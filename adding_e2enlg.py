import os

import evaluate
import numpy as np
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (GPT2LMHeadModel, GPT2Tokenizer, Trainer,
                          TrainingArguments)

if not os.path.exists("models"):
        os.mkdir("models")

def simple_tokenization(examples, tokenizer):
    return tokenizer(examples["human_reference"])

# copied from training_ensemble.py
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

def init_training():
    tokenizer = GPT2Tokenizer.from_pretrained("GPT2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # e2e does not seem to have a subset
    dataset = load_dataset("e2e_nlg")
    remove_columns = ['meaning_representation', 'human_reference']
    tokenized_dataset = dataset.map(simple_tokenization, fn_kwargs={"tokenizer": tokenizer}, num_proc=4, batched=True, remove_columns=remove_columns)
    lm_dataset = tokenized_dataset.map(
        group_texts,
        fn_kwargs={"block_size": 512},
        batched=True,
        num_proc=4
    )
        
    pretrained_model = GPT2LMHeadModel.from_pretrained("GPT2",
                        pad_token_id=tokenizer.eos_token_id)
    lora_config = LoraConfig(
        r=4,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    lora_model = get_peft_model(pretrained_model, lora_config)

    return lm_dataset, lora_model

def train_model():
    lm_dataset, lora_model = init_training()

    train_args = TrainingArguments(
        output_dir="models",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=1,
        learning_rate=2e-4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        per_device_train_batch_size=8,
        warmup_steps=500,
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        print(predictions)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=lora_model,
        args=train_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    train_model()