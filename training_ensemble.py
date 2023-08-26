import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus import PrivacyEngine
import os
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer, set_seed
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import argparse

set_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="GPT2")
parser.add_argument("--dataset", type=str, default="wikitext")
parser.add_argument("--subset", type=str, default="wikitext-103-raw-v1")
parser.add_argument("--num_ensemble", type=int, default=8)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--lora_r", type=int, default=4)
parser.add_argument("--lora_alpha", type=int, default=32)
parser.add_argument("--lora_dropout", type=float, default=0.1)
parser.add_argument("--block_size", type=int, default=512)
parser.add_argument("--learning_rate", type=float, default=2e-4)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--training_type", type=str, default="sub-samp-and-agg")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--noise_multiplier", type=float, default=1.)
parser.add_argument("--max_grad_norm", type=float, default=1.)
parser.add_argument("--epsilon", type=float, default=1.)

args = parser.parse_args()

START = 7 
if not os.path.exists("models"):
    os.mkdir("models")
if args.num_ensemble == 1:
    model_dir = "models"
else:
    model_dir = os.path.join("models", f"{args.num_ensemble}_ensemble")

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

def init_training():
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    pretrained_model = GPT2LMHeadModel.from_pretrained(args.model_name,
                        pad_token_id=tokenizer.eos_token_id)
    dataset = load_dataset(args.dataset, args.subset)
    tokenize_function = wiki_tokenize_function if args.dataset == "wikitext" else None
    remove_columns = ["text"] if args.dataset == "wikitext" else None
    tokenized_dataset = dataset.map(tokenize_function,
                                    fn_kwargs={"tokenizer": tokenizer},
                                    batched=True,
                                    num_proc=4,
                                    remove_columns=remove_columns
                                    )
    lm_dataset = tokenized_dataset.map(
        group_texts,
        fn_kwargs={"block_size": args.block_size},
        batched=True,
        num_proc=4
    ) 
    return lm_dataset, tokenizer, pretrained_model

def train_ensemble():
    lm_dataset, tokenizer, pretrained_model = init_training()
    for i in range(START, args.num_ensemble):
        lm_shards = {} 
        if args.num_ensemble == 1:
            lm_shards['train'] = lm_dataset['train']
            lm_shards['validation'] = lm_dataset['validation']
        else:
            lm_shards['train'] = lm_dataset['train'].shard(num_shards=args.num_ensemble, index=i)
            lm_shards['validation'] = lm_dataset['validation'].shard(num_shards=args.num_ensemble, index=i)

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        lora_model = get_peft_model(pretrained_model, lora_config)

        print(f"\n\nTraining Shard {i} of size {len(lm_shards['train'])}")
        print(f"Trainable paramters {print_trainable_parameters(lora_model)}\n\n")

        output_dir = 0
        if args.num_ensemble == 1:
            output_dir = os.path.join(model_dir, f"lora-{args.model_name}-finetuned-{args.subset}")
        else:
            output_dir = os.path.join(model_dir,
                                    f"lora-{args.model_name}-{i}-finetuned-{args.subset}")

        train_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            load_best_model_at_end=True,
            per_device_train_batch_size=args.batch_size,
        )
        train_args = train_args.set_lr_scheduler(name="linear", warmup_steps=500)
        trainer = Trainer(
            model=lora_model,
            args=train_args,
            train_dataset=lm_shards['train'],
            eval_dataset=lm_shards['validation'],
        )
        trainer.train()
        eval_results = trainer.evaluate()
        print(f"\n\nPerplexity: {math.exp(eval_results['eval_loss']):.2f}\n\n")
        trainer.save_model(output_dir)

def wiki_tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"])


def group_texts(examples, block_size):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def init_dp_training(rank):
    lm_dataset, tokenizer, pretrained_model = init_training()
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    lora_model = get_peft_model(pretrained_model, lora_config)
    model = DPDDP(lora_model)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    train_data_loader = DataLoader(
        lm_dataset['train'],
        batch_size=args.batch_size
    )

    val_data_loader = DataLoader(
        lm_dataset['validation'],
        batch_size=args.batch_size
    )

    privacy_engine = PrivacyEngine(accountant="rdp")
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        target_epsilon=args.epsilon,
        target_delta=0,
        epochs=args.epochs
        max_grad_norm=args.max_grad_norm,
        #noise_multiplier=args.noise_multiplier,
    )
    return model, optimizer, train_data_loader, val_data_loader, privacy_engine

def dpsgd(rank, world_size):
    setup(rank, world_size)
    model, optimizer, train_data_loader, val_data_loader, privacy_engine = init_dp_training(rank)
    criterion = nn.CrossEntropyLoss()
    model.to(rank)
    model.train()

    for e in range(args.epochs):
        losses = []
        correct = 0
        total = 0
        for data, target in train_data_loader:
            data, target = data.to(rank), target
            optimizer.zero_grad()
            output = model(data)

            pred = output.argmax(dim=1, keepdim=True)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        test_accuracy = evaluate(model, val_data_loader, rank)
        train_accuracy = correct / total
        epsilon = privacy_engine.get_epsilon()
        
        if rank == 0:
            print(
                f"Epoch: {e} \t"
                f"Train Loss: {np.mean(losses):.4f} | "
                f"Train Accuracy: {train_accuracy:.2f} | "
                f"Test Accuracy: {test_accuracy:.2f} |"
                f"(ε = {epsilon:.2f})"
            )

    cleanup()    

def accuracy(preds, labels):
    return (preds == labels).mean()

# define evaluation cycle
def evaluate(model, test_dataloader, device):    
    model.eval()

    loss_arr = []
    accuracy_arr = []
    
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3]}

            outputs = model(**inputs)
            loss, logits = outputs[:2]
            
            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
            labels = inputs['labels'].detach().cpu().numpy()
            
            loss_arr.append(loss.item())
            accuracy_arr.append(accuracy(preds, labels))
    
    model.train()
    return np.mean(loss_arr), np.mean(accuracy_arr) 

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
    f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

world_size = torch.cuda.device_count()

if __name__ == "__main__":
    if args.training_type == "sub-samp-and-agg":
        train_ensemble()
    elif args.training_type == "dpsgd":
        mp.spawn(
            dpsgd,
            args=(world_size),
            nprocs=world_size,
            join=True
        )
    else:
        raise ValueError("Incorrect training")