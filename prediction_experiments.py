#!venv/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import os 
import argparse
from ensemble import Ensemble
from datasets import load_dataset
from training_ensemble import group_texts, wiki_tokenize_function
from peft import PeftModel
import copy
import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import bisect, minimize 
import numpy as np
import math

def main(args):
    #alpha = math.ceil(4 * np.log(1/args.delta) / (3*args.epsilon) + 1)
    alpha = args.alpha
    epsilon = args.epsilon - np.log((alpha-1)/alpha) + (np.log(args.delta) + np.log(alpha))/(alpha-1)
    #print("Alpha", alpha)
    #print("Epsilon", epsilon)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    pub_model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                    pad_token_id=tokenizer.eos_token_id).to(
                                                    args.device)
    model_dir = os.path.join("models", f"{args.num_ensemble}_ensemble")

    model_paths = [os.path.join(model_dir, f"lora-{args.model_name}-{i}-finetuned-{args.data_subset}")
                    for i in range(args.num_ensemble)]
    priv_ensemble = Ensemble(model_paths,
                             args.model_name,
                             tokenizer,
                             args.device,
                             q_budget=args.query_budget,
                             alpha=alpha,
                             eps=epsilon,
                             delta=args.delta,
                             p=args.p)

    fine_tuned_model_dir = os.path.join("models", f"lora-{args.model_name}-finetuned-{args.data_subset}")
    fine_tuned_model = PeftModel.from_pretrained(copy.deepcopy(pub_model),
                                                 fine_tuned_model_dir,
                                                 pad_token_id=tokenizer.eos_token_id).to(
                                                 args.device)
    dp_fine_tuned_model = torch.load(os.path.join("models", f"lora-{args.model_name}-8.0-dp-finetuned-{args.data_subset}.pt")).to(args.device)
    #dp_fine_tuned_model = torch.load(os.path.join("models", f"lora-{args.model_name}-{args.epsilon}-dp-finetuned-{args.data_subset}.pt")).to(args.device)

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

    test_data = lm_dataset['test']
    test_data.set_format(type="torch")

    pub_neg_log_likelihood = []
    fine_tuned_neg_log_likelihood = []
    dp_fine_tuned_neg_log_likelihood = []
    ensemble_neg_log_likelihood= []
    step_size = args.query_budget // args.seq_length
    test_loader = DataLoader(test_data.select([i + args.start for i in range(args.start+step_size)]))#, shuffle=True)
    fine_tuned_model.eval()
    k = 0
    for i, data in enumerate(test_loader):
        labels = data['labels'].to(args.device)
        input_ids = data['input_ids'].to(args.device)
        with torch.no_grad():
            pub_output_logits = pub_model(input_ids).logits 
            fine_tuned_output_logits =fine_tuned_model(input_ids).logits 
            dp_fine_tuned_output_logits = dp_fine_tuned_model(input_ids).logits 
            output_dists = priv_ensemble.pred_dist(input_ids)
            ensemble_logits = []

            if k < args.query_budget:
                for j in range(seq_length):
                    token_softmax = [output_dist[j] for output_dist in output_dists]
                    ensemble_output_dist = priv_ensemble.priv_pred(token_softmax)
                    ensemble_logits.append(torch.log(ensemble_output_dist.cpu()))
                    k += 1

                ensemble_logits = torch.stack(ensemble_logits)
                ensemble_neg_log_likelihood.append(calc_loss(ensemble_logits, labels.cpu()))
            else:
                break

        pub_neg_log_likelihood.append(calc_loss((pub_output_logits), labels))
        fine_tuned_neg_log_likelihood.append(calc_loss((fine_tuned_output_logits), labels))
        dp_fine_tuned_neg_log_likelihood.append(calc_loss(dp_fine_tuned_output_logits, labels))

    pre_trained_ppl = torch.exp(torch.stack(pub_neg_log_likelihood))
    fine_tuned_ppl = torch.exp(torch.stack(fine_tuned_neg_log_likelihood))
    dp_fine_tuned_ppl = torch.exp(torch.stack(dp_fine_tuned_neg_log_likelihood))
    ensemble_ppl = torch.exp(torch.stack(ensemble_neg_log_likelihood))

    #print(f"Perplexity score for Pre-Trained Model: {pre_trained_ppl.mean():.2f}")
    #print(f"Perplexity score for Fine-Tuned Model: {fine_tuned_ppl.mean():.2f}")
    #print(f"Perplexity score for DP-Fine-Tuned Model: {dp_fine_tuned_ppl.mean():.2f}")
    #print(f"Perplexity score for Ensemble Private Prediction Model: {ensemble_ppl.mean():.2f}")

    #priv_ensemble.print_priv_losses()
    priv_ensemble.print_lambdas()
    #priv_ensemble.plot_individual_loss()
    priv_ensemble.plot_lambdas()

    return pre_trained_ppl.mean().cpu(), fine_tuned_ppl.mean().cpu(), dp_fine_tuned_ppl.mean().cpu(), ensemble_ppl.mean().cpu()

def calc_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = nn.CrossEntropyLoss()
    return loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

if __name__ == "__main__":
    set_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ensemble", type=int, default=8)
    parser.add_argument("--model_name", type=str, default="GPT2")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--data_subset", type=str, default="wikitext-103-v1")
    parser.add_argument("--device", type=str, default="cuda:6")
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--query_budget", type=int, default=1024)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--alpha", type=int, default=2)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    pub_ppl_list = []
    ft_ppl_list = []
    dpsgd_ppl_list = []
    ensemble_ppl_list = []
    step_size = args.query_budget // args.seq_length
    for i in range(0, args.iters):
        args.start = i * step_size 
        pub_ppl, ft_ppl, dpsgd_ppl, ensemble_ppl = main(args)
        pub_ppl_list.append(pub_ppl)
        ft_ppl_list.append(ft_ppl)
        dpsgd_ppl_list.append(dpsgd_ppl)
        ensemble_ppl_list.append(ensemble_ppl)

    print(f"{np.mean(pub_ppl_list):.2f}")
    print(f"{np.mean(ft_ppl_list):.2f}")
    print(f"{np.mean(dpsgd_ppl_list):.2f}")
    print(f"{np.mean(ensemble_ppl_list):.2f}")