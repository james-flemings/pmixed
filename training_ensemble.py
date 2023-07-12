import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import math

from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer, set_seed
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import argparse

set_seed(0)

#dist.init_process_group("gloo")
#world_size = dist.get_world_size()
#rank = dist.get_rank()

#DEVICE = f"cuda:{rank}"

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="GPT2")
parser.add_argument("--dataset", type=str, default="wikitext")
parser.add_argument("--subset", type=str, default="wikitext-103-raw-v1")
parser.add_argument("--num_ensemble", type=int, default=8)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--block_size", type=int, default=526)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--weight_decay", type=float, default=0.01)




def main():
    args = parser.parse_args()
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    pretrained_model = GPT2LMHeadModel.from_pretrained(args.model_name,
                        pad_token_id=tokenizer.eos_token_id)#.to(DEVICE)

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

    def accuracy(preds, labels):
        return (preds == labels).mean()

    if not os.path.exists("models"):
        os.mkdir("models")

    print(f"\n\nTotal size of training dataset {len(lm_dataset['train'])}\n\n")

    for i in range(8):
        lm_shards = {} 
        lm_shards['train'] = lm_dataset['train'].shard(num_shards=args.num_ensemble, index=i)
        lm_shards['validation'] = lm_dataset['train'].shard(num_shards=args.num_ensemble, index=i)
        print(f"\n\nTraining Shard {i} of size {len(lm_shards['train'])}\n\n")

        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM"
        )
        lora_model = get_peft_model(pretrained_model, lora_config)#.to(DEVICE)
        #lora_model = DDP(lora_model, device_ids=[rank])

        output_dir = os.path.join("models", f"lora-{args.model_name}-{i}-finetuned-wikitext2")
        train_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        trainer = Trainer(
            model=lora_model,
            args=train_args,
            train_dataset=lm_shards['train'],
            eval_dataset=lm_shards['validation'],
        )
        trainer.train()
        eval_results = trainer.evaluate()
        print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


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
    main()