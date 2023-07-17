import torch
import torch.nn as nn
from torcheval.metrics.text import Perplexity
from transformers import AutoModelForCausalLM, AutoTokenizer
import os 
import argparse
from ensemble import Ensemble
from datasets import load_dataset
from training_ensemble import group_texts, wiki_tokenize_function


parser = argparse.ArgumentParser()
parser.add_argument("--num_ensemble", type=int, default=8)
parser.add_argument("--model_name", type=str, default="GPT2")
parser.add_argument("--dataset", type=str, default="wikitext")
parser.add_argument("--data_subset", type=str, default="wikitext-103-raw-v1")
parser.add_argument("--device", type=str, default="cuda:7")
parser.add_argument("--seq_length", type=int, default=512)
parser.add_argument("--query_budget", type=int, default=1024)

def main():
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    pub_model = AutoModelForCausalLM.from_pretrained(args.model_name, pad_token_id=tokenizer.eos_token_id).to(args.device)
    
    seq_length = 512
    dataset = load_dataset(args.dataset, args.data_subset)

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
        fn_kwargs={"block_size": seq_length},
        batched=True,
        num_proc=4
    ) 

    test_data = lm_dataset['validation']
    test_data.set_format(type="torch")

    model_paths = [os.path.join("models", f"lora-{args.model_name}-{i}-finetuned-{args.data_subset}")
                    for i in range(args.num_ensemble)]
    priv_ensemble = Ensemble(model_paths,
                             pub_model,
                             args.device,
                             q_budget=args.query_budget)

    pub_neg_log_likelihood = []
    fine_tuned_neg_log_likelihood = []
    for i in range(args.query_budget // seq_length):
        data = test_data[i]['input_ids'].to(args.device)
        labels = test_data[i]['labels'].to(args.device)
        with torch.no_grad():
            pub_output = pub_model(data, labels=labels)
            pub_output_logits = pub_output.logits
            
            output_dists = priv_ensemble.pred_dist(data)
            fine_tuned_logits = []

            for j in range(seq_length):
                token_softmax = [output_dist[j] for output_dist in output_dists]
                fine_tuned_dist = priv_ensemble.reg_pred(token_softmax)
                fine_tuned_logits.append(torch.log(fine_tuned_dist))

        pub_neg_log_likelihood.append(calc_loss(pub_output_logits, labels))
        fine_tuned_logits = torch.stack(fine_tuned_logits)
        fine_tuned_neg_log_likelihood.append(calc_loss(fine_tuned_logits, labels))

    pre_trained_ppl = torch.exp(torch.stack(pub_neg_log_likelihood)).mean()
    fine_tuned_ppl = torch.exp(torch.stack(fine_tuned_neg_log_likelihood)).mean()
    print(pre_trained_ppl)
    print(fine_tuned_ppl)

def calc_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = nn.CrossEntropyLoss()
    return loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

if __name__ == "__main__":
    main()