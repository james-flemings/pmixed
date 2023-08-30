import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import os 
import argparse
from ensemble import Ensemble
from datasets import load_dataset
from training_ensemble import group_texts, wiki_tokenize_function
from peft import PeftModel
import copy
import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import bisect
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--num_ensemble", type=int, default=8)
parser.add_argument("--model_name", type=str, default="GPT2")
parser.add_argument("--dataset", type=str, default="wikitext")
parser.add_argument("--data_subset", type=str, default="wikitext-103-raw-v1")
parser.add_argument("--device", type=str, default="cuda:6")
parser.add_argument("--seq_length", type=int, default=512)
parser.add_argument("--query_budget", type=int, default=1024)
parser.add_argument("--target_multiplier", type=float, default=1.0)
parser.add_argument("--epsilon", type=float, default=1.0)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--p_value", type=float, default=1.0)

def main():
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_dir = os.path.join("models", f"{args.num_ensemble}_ensemble")

    model_paths = [os.path.join(model_dir, f"lora-{args.model_name}-{i}-finetuned-{args.data_subset}")
                    for i in range(args.num_ensemble)]
    #priv_ensemble = Ensemble(model_paths,
    #                         args.model_name,
    #                         tokenizer,
    #                         args.device,
    #                         q_budget=args.query_budget,
    #                         eps=args.epsilon,
    #                         target_mult=args.target_multiplier)

    pub_model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                    pad_token_id=tokenizer.eos_token_id).to(
                                                    args.device)

    fine_tuned_model_dir = os.path.join("models", f"lora-{args.model_name}-finetuned-{args.data_subset}")
    fine_tuned_model = PeftModel.from_pretrained(copy.deepcopy(pub_model),
                                                 fine_tuned_model_dir,
                                                 pad_token_id=tokenizer.eos_token_id).to(
                                                 args.device)
    #dp_fine_tuned_model = torch.load(os.path.join("models", f"lora-{args.model_name}-{args.epsilon}-dp-finetuned-{args.data_subset}.pt")).to(args.device)
    dp_fine_tuned_model = torch.load(os.path.join("models", f"lora-{args.model_name}-1.0-dp-finetuned-{args.data_subset}.pt")).to(args.device)

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
    priv_neg_log_likelihood= []
    test_loader = DataLoader(test_data)
    priv_loss = []
    lambdas = []
    left_over = 0
    target = args.epsilon / args.query_budget

    for i, data in enumerate(tqdm.tqdm(test_loader)):
        labels = data['labels'].to(args.device)
        input_ids = data['input_ids'].to(args.device)
        with torch.no_grad():
            #pub_output_logits = top_p_filtering(pub_model(input_ids).logits.squeeze(), 0.95).unsqueeze(0)
            pub_output_logits =pub_model(input_ids).logits
            #fine_tuned_output_logits = top_p_filtering(fine_tuned_model(input_ids).logits.squeeze().cpu(), 0.95).unsqueeze(0)
            fine_tuned_output_logits =fine_tuned_model(input_ids).logits
            dp_fine_tuned_output_logits = dp_fine_tuned_model(input_ids).logits
            fine_tuned_output_softmax = nn.functional.softmax(fine_tuned_output_logits.squeeze())
            pub_output_logits_filtered = top_p_filtering(fine_tuned_output_logits.clone().squeeze(), args.p_value) 
            pub_output_softmax = nn.functional.softmax(pub_output_logits_filtered/args.temperature)
            
            #output_dists = priv_ensemble.pred_dist(input_ids)
            ensemble_logits = []
            priv_logits = []

            if i < args.query_budget // seq_length:
                for j in range(seq_length):
                    #token_softmax = [output_dist[j] for output_dist in output_dists]
                    #ensemble_output_dist = priv_ensemble.priv_pred(token_softmax)
                    #ensemble_logits.append(torch.log(ensemble_output_dist))
                    priv_pred, loss, lambd = private_pred(fine_tuned_output_softmax[j], pub_output_softmax[j],
                                             (target + left_over))
                    left_over += (target - loss)
                    priv_logits.append(torch.log(priv_pred))
                    priv_loss.append(loss)
                    lambdas.append(lambd)

                #ensemble_logits = torch.stack(ensemble_logits)
                #ensemble_neg_log_likelihood.append(calc_loss(ensemble_logits, labels))
                priv_logits = torch.stack(priv_logits)
                priv_neg_log_likelihood.append(calc_loss(priv_logits, labels))
            else:
                break

        pub_neg_log_likelihood.append(calc_loss((pub_output_logits), labels))
        fine_tuned_neg_log_likelihood.append(calc_loss((fine_tuned_output_logits), labels))
        dp_fine_tuned_neg_log_likelihood.append(calc_loss(dp_fine_tuned_output_logits, labels))

    pre_trained_ppl = torch.exp(torch.stack(pub_neg_log_likelihood))
    fine_tuned_ppl = torch.exp(torch.stack(fine_tuned_neg_log_likelihood))
    dp_fine_tuned_ppl = torch.exp(torch.stack(dp_fine_tuned_neg_log_likelihood))
    #ensemble_ppl = torch.exp(torch.stack(ensemble_neg_log_likelihood))
    priv_ppl = torch.exp(torch.stack(priv_neg_log_likelihood))

    print(f"Perplexity score for Pre-Trained Model: {pre_trained_ppl.mean():.2f}")
    print(f"Perplexity score for Fine-Tuned Model: {fine_tuned_ppl.mean():.2f}")
    print(f"Perplexity score for DP-Fine-Tuned Model: {dp_fine_tuned_ppl.mean():.2f}")
    print(f"Perplexity score for Private Prediction Model: {priv_ppl.mean():.2f}")

    print(f"Total privacy loss of model: {np.sum(priv_loss):.4f}")
    print(f"Average lambda value: {np.mean(lambdas):.4f}")
    print(f"Min lambda: {np.min(lambdas)}")
    print(f"Max lambda: {np.max(lambdas)}")

    #priv_ensemble.print_priv_losses()
    #priv_ensemble.print_lambdas()
    #priv_ensemble.plot_individual_loss()
    #priv_ensemble.plot_lambdas()
    #plot_ppl(ensemble_ppl)

def calc_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = nn.CrossEntropyLoss()
    return loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

def plot_ppl(ppl_scores):
    x = [i for i in range(len(ppl_scores))]
    plt.plot(x, ppl_scores.cpu())
    plt.savefig(os.path.join("plt", "ensemble_perplexity_scores.png"))
    plt.clf()

def private_pred(priv_model, pub_model, target):
    #lambd = lambda_solver_bisection(pub_model.cpu(), target)
    i = torch.min(torch.nonzero(pub_model))
    lambd = max((1 - 1/(pub_model[i]*(np.exp(target)-1) + 1)).cpu(), 0)
    #lambd = 0.01 
    loss = calc_indiv_priv_loss(pub_model, lambd)
    pred = lambd * priv_model + (1-lambd) * pub_model + 1e-20
    return pred, loss.cpu(), lambd

def lambda_solver_bisection(pub_pred, target):
    def f(lambd):
        eps = calc_indiv_priv_loss(pub_pred, lambd)
        return (eps - target)

    if f(0.99) <= 0.0:
        lambd = 0.99 
    else:
        lambd = bisect(f, 0, 0.99, maxiter=10, disp=False)
    return lambd

def calc_indiv_priv_loss(pub_pred, lambd):
    i = torch.min(torch.nonzero(pub_pred))
    loss = torch.log(lambd / ((1-lambd) * pub_pred[i]) + 1)
    return loss

def renyiDiv(p, q, alpha=float('inf')):
    if alpha == float('inf'):
        RD = torch.log(torch.max(p/q))
    elif alpha == 1:
        RD = torch.sum(p*torch.log(p/q))
    else:
        RD = 1/(alpha-1)*torch.log(
            torch.sum((p**alpha)/(q**(alpha-1))))
    if torch.isnan(RD):
        RD = torch.log(torch.max(p/q))
    return RD 

def top_p_filtering(logits, p, filter_value=-float("Inf")):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs > p

    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = filter_value

    return logits

if __name__ == "__main__":
    main()