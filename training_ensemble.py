import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.optimizers import DistributedDPOptimizer
from opacus.data_loader import DPDataLoader
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus import PrivacyEngine
import os
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer, set_seed
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import argparse
import tqdm

START = 0 
def init_training(args):
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    pretrained_model = GPT2LMHeadModel.from_pretrained(args.model_name,
                        pad_token_id=tokenizer.eos_token_id)
    if args.subset == "None":
        args.subset = None
    dataset = load_dataset(args.dataset, args.subset)
    #remove_columns = ["text"] if args.dataset == "wikitext" else None
    remove_columns = ['text']
    tokenized_dataset = dataset.map(tokenize_function,
                                    fn_kwargs={"tokenizer": tokenizer},
                                    batched=True,
                                    num_proc=args.num_proc,
                                    remove_columns=remove_columns
                                    )
    lm_dataset = tokenized_dataset.map(
        group_texts,
        fn_kwargs={"block_size": args.block_size},
        batched=True,
        num_proc=args.num_proc
    ) 
    return lm_dataset, tokenizer, pretrained_model

def train_ensemble(args, model_dir):
    lm_dataset, tokenizer, pretrained_model = init_training(args)
    for i in range(START, args.num_ensemble):
        lm_shards = {} 
        if args.num_ensemble == 1:
            lm_shards['train'] = lm_dataset['train']
            if args.dataset == 'wikitext':
                lm_shards['validation'] = lm_dataset['validation']
            else:
                lm_shards['validation'] = None
        else:
            lm_shards['train'] = lm_dataset['train'].shard(num_shards=args.num_ensemble, index=i)
            if args.dataset == 'wikitext':
                lm_shards['validation'] = lm_dataset['validation'].shard(num_shards=args.num_ensemble, index=i)
            else:
                lm_shards['validation'] = None

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        lora_model = get_peft_model(pretrained_model, lora_config)

        #print(f"\n\nTraining Shard {i} of size {len(lm_shards['train'])}")
        #print_trainable_parameters(lora_model)

        output_dir = 0
        if args.num_ensemble == 1:
            output_dir = os.path.join(model_dir, f"lora-{args.model_name}-finetuned-{args.dataset}")
        else:
            output_dir = os.path.join(model_dir,
                                    f"lora-{args.model_name}-{i}-finetuned-{args.dataset}")
        eval_strat = 'no' if args.dataset == 'lm1b' else 'epoch'
        train_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy=eval_strat,
            save_strategy=eval_strat,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            load_best_model_at_end=True,
            per_device_train_batch_size=args.batch_size,
            lr_scheduler_type="linear",
            warmup_steps=500,
        )
        #train_args = train_args.set_lr_scheduler(name="linear", warmup_steps=500)
        trainer = Trainer(
            model=lora_model,
            args=train_args,
            train_dataset=lm_shards['train'],
            eval_dataset=lm_shards['validation'],
        )
        trainer.train()
        #eval_results = trainer.evaluate()
        #print(f"\n\nPerplexity: {math.exp(eval_results['eval_loss']):.2f}\n\n")
        trainer.save_model(output_dir)

def tokenize_function(examples, tokenizer):
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

def init_dp_training(rank, args):
    lm_dataset, tokenizer, pretrained_model = init_training(args)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    lora_model = get_peft_model(pretrained_model, lora_config)
    model = DPDDP(lora_model)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                           weight_decay=args.weight_decay)
    lm_dataset['train'].set_format(type='torch')
    if args.dataset == 'wikitext':
        lm_dataset['validation'].set_format(type='torch')

    train_data_loader = DataLoader(
        lm_dataset['train'],
        batch_size=args.dp_batch_size
    )

    val_data_loader = None
    if args.dataset == 'wikitext':
        val_data_loader = DataLoader(
            lm_dataset['validation'],
            batch_size=args.dp_batch_size
        )

    privacy_engine = PrivacyEngine(accountant="rdp")
    model, optimizer, train_data_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_data_loader,
        target_epsilon=args.epsilon,
        target_delta=args.delta,
        epochs=args.epochs,
        max_grad_norm=args.max_grad_norm,
        #noise_multiplier=args.noise_multiplier,
    )
    return model, optimizer, train_data_loader, val_data_loader, privacy_engine

def dpsgd(rank, world_size, args, model_dir):
    setup(rank, world_size)
    model, optimizer, train_data_loader, val_data_loader, privacy_engine = init_dp_training(rank, args)
    model.to(rank)
    model.train()

    for e in range(args.epochs):
        losses = []
        with BatchMemoryManager(
        data_loader=train_data_loader, 
        max_physical_batch_size=args.batch_size, 
        optimizer=optimizer
        ) as memory_safe_data_loader:
            for data in tqdm.tqdm(memory_safe_data_loader):
                input_ids, labels = data['input_ids'].to(rank), data['labels'].to(rank)
                optimizer.zero_grad()
                output = model(input_ids, labels=labels)

                loss, logits = output[0], output[1]
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            #val_losses = evaluate(model, val_data_loader, rank, criterion)
            epsilon = privacy_engine.get_epsilon(delta=args.delta)
            
            if rank == 0:
                print(
                    f"Epoch: {e} \t"
                    f"Train Loss: {np.mean(losses):.4f} | "
                    #f"Validation Loss: {np.mean(val_losses):.4f} | "
                    f"(ε = {epsilon:.2f})"
                )
            output_dir = os.path.join(model_dir, f"lora-{args.model_name}-{args.epsilon}-dp-finetuned-{args.dataset}.pt")
            torch.save(model._module, output_dir)
    cleanup()    

def accuracy(preds, labels):
    return (preds == labels).mean()

# define evaluation cycle
def evaluate(model, test_dataloader, device, criterion):    
    model.eval()
    losses = []

    with torch.no_grad():
        for data in test_dataloader:
            input_ids, labels = data['input_ids'].to(device), data['labels'].to(device)
            output = model(input_ids, labels=labels)
            losses.append(output[0].item())

    model.train()
    return losses

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

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

if __name__ == "__main__":
    set_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="GPT2")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--subset", type=str, default="wikitext-103-raw-v1")
    parser.add_argument("--num_ensemble", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dp_batch_size", type=int, default=32)
    parser.add_argument("--training_type", type=str, default="sub-samp-and-agg")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--noise_multiplier", type=float, default=1.)
    parser.add_argument("--max_grad_norm", type=float, default=1.)
    parser.add_argument("--epsilon", type=float, default=1.)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_proc", type=int, default=64)
    args = parser.parse_args()
    world_size = args.num_gpus

    if not os.path.exists("models"):
        os.mkdir("models")
    if args.num_ensemble == 1 or args.training_type == "dpsgd":
        model_dir = "models"
    else:
        model_dir = os.path.join("models", f"{args.num_ensemble}_ensemble")

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    if args.training_type == "sub-samp-and-agg":
        train_ensemble(args, model_dir)
    elif args.training_type == "dpsgd":
        mp.spawn(
            dpsgd,
            args=(world_size, args, model_dir),
            nprocs=world_size,
            join=True
        )
    else:
        raise ValueError("Incorrect training")