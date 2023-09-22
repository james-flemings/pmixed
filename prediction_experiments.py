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
    set_seed(args.seed)
    #alpha = math.ceil(4 * np.log(1/args.delta) / (3*args.epsilon) + 1)
    alpha = args.alpha
    epsilon = args.epsilon - np.log(1/args.delta)/(args.alpha-1)
    print("Alpha", alpha)
    print("Epsilon", epsilon)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    pub_model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                    pad_token_id=tokenizer.eos_token_id).to(
                                                    args.device)
    model_dir = os.path.join("models", f"{args.num_ensemble}_ensemble")

    model_paths = [os.path.join(model_dir, f"lora-{args.model_name}-{i}-finetuned-wikitext-103-raw-v1")
                    for i in range(args.num_ensemble)]
    priv_ensemble = Ensemble(model_paths,
                             args.model_name,
                             tokenizer,
                             args.device,
                             q_budget=args.query_budget,
                             alpha=alpha,
                             eps=epsilon,
                             target_mult=args.target_multiplier,
                             p_value=args.p_value)

    fine_tuned_model_dir = os.path.join("models", f"lora-{args.model_name}-finetuned-{args.data_subset}")#/checkpoint-3582")
    fine_tuned_model = PeftModel.from_pretrained(copy.deepcopy(pub_model),
                                                 fine_tuned_model_dir,
                                                 pad_token_id=tokenizer.eos_token_id).to(
                                                 args.device)
    #dp_fine_tuned_model = torch.load(os.path.join("models", f"lora-{args.model_name}-6.0-dp-finetuned-{args.data_subset}.pt")).to(args.device)
    dp_fine_tuned_model = torch.load(os.path.join("models", f"lora-{args.model_name}-{args.epsilon}-dp-finetuned-{args.data_subset}.pt")).to(args.device)

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
    priv_neg_log_likelihood= []
    ensemble_neg_log_likelihood= []

    test_loader = DataLoader(test_data, shuffle=True)
    priv_loss = []
    lambdas = []
    fine_tuned_model.eval()

    for i, data in enumerate(test_loader):
        labels = data['labels'].to(args.device)
        input_ids = data['input_ids'].to(args.device)
        with torch.no_grad():
            pub_output_logits = pub_model(input_ids).logits 
            fine_tuned_output_logits =fine_tuned_model(input_ids).logits 
            dp_fine_tuned_output_logits = dp_fine_tuned_model(input_ids).logits 

            fine_tuned_output_softmax = nn.functional.softmax(fine_tuned_output_logits.squeeze(), dim=-1)
            pub_output_softmax = nn.functional.softmax(pub_output_logits.squeeze(), dim=-1)

            output_dists = priv_ensemble.pred_dist(input_ids)
            ensemble_logits = []
            priv_logits = []

            if i < args.query_budget // seq_length:
                for j in tqdm.tqdm(range(seq_length)):
                    priv_pred, loss, lambd  = private_pred(fine_tuned_output_softmax[j], pub_output_softmax[j],
                                             epsilon, args.query_budget, alpha, args.device)

                    token_softmax = [output_dist[j] for output_dist in output_dists]
                    ensemble_output_dist = priv_ensemble.priv_pred(token_softmax)
                    ensemble_logits.append(torch.log(ensemble_output_dist.cpu()))
                    priv_logits.append(torch.log(priv_pred))
                    priv_loss.append(loss)
                    lambdas.append(lambd)

                priv_logits = torch.stack(priv_logits)
                priv_neg_log_likelihood.append(calc_loss(priv_logits, labels))
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
    priv_ppl = torch.exp(torch.stack(priv_neg_log_likelihood))
    ensemble_ppl = torch.exp(torch.stack(ensemble_neg_log_likelihood))

    print(f"Perplexity score for Pre-Trained Model: {pre_trained_ppl.mean():.2f}")
    print(f"Perplexity score for Fine-Tuned Model: {fine_tuned_ppl.mean():.2f}")
    print(f"Perplexity score for DP-Fine-Tuned Model: {dp_fine_tuned_ppl.mean():.2f}")
    print(f"Perplexity score for Private Prediction Model: {priv_ppl.mean():.2f}")
    print(f"Perplexity score for Ensemble Private Prediction Model: {ensemble_ppl.mean():.2f}")

    print(f"Total privacy loss of model: {sum(priv_loss):.4f}")
    print(f"Max loss", max(priv_loss))
    print(f"Average lambda value: {np.mean(lambdas):.4f}")
    print(f"Min lambda: {np.min(lambdas)}")
    print(f"Max lambda: {np.max(lambdas)}")

    priv_ensemble.print_priv_losses()
    priv_ensemble.print_lambdas()
    priv_ensemble.plot_individual_loss()
    priv_ensemble.plot_lambdas()

    return pre_trained_ppl.mean().cpu(), fine_tuned_ppl.mean().cpu(), dp_fine_tuned_ppl.mean().cpu(), priv_ppl.mean().cpu(), ensemble_ppl.mean().cpu()

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

def private_pred(priv_model, pub_model, epsilon, budget, alpha=2, device="cpu"):
    #lambd = lambda_solver_bisection(priv_model.cpu(), pub_model.cpu(), epsilon, budget, alpha)
    lambd = 0.1
    pred = lambd * priv_model + (1-lambd) * pub_model
    loss = min(data_independent_loss(pred, pub_model, alpha),
               data_dependent_loss(pred.cpu(), pub_model, alpha))
    return pred, loss, lambd

def lambda_solver_bisection(p_priv, p_pub, epsilon, budget, alpha):
    def f(lambd):
        p_mix = lambd * p_priv + (1-lambd) * p_pub
        eps = min(data_independent_loss(p_mix, p_pub, alpha), data_dependent_loss(p_mix, p_pub, alpha))
        return (eps - epsilon/budget)
    if f(1) <= 0.0:
        lambd = 1 
    else:
        lambd = bisect(f, 0, 1, maxiter=20, disp=False)
    return lambd

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

def data_independent_loss(p_mix, p_pub, alpha):
    #return 2 * alpha * max(renyiDiv(p_mix.cpu(), p_pub.cpu(), alpha=2*alpha),
    #           renyiDiv(p_pub.cpu(), p_mix.cpu(), alpha=2*alpha)).item()
    return alpha * max(renyiDiv(p_mix.cpu(), p_pub.cpu(), alpha=float('Inf')),
               renyiDiv(p_pub.cpu(), p_mix.cpu(), alpha=float("Inf"))).item()**2

def data_dependent_loss(p_mix, p_pub, alpha):
    epsilon = data_independent_loss(p_mix, p_pub, alpha)
    if epsilon == 0:
        epsilon = 1 / 1024 
    q = 1 - torch.max(p_mix)

    alpha_2 = torch.ceil(np.sqrt(1/epsilon * torch.log(1/q)))
    alpha_1 = 1 + alpha_2

    epsilon_1 = epsilon * alpha_1
    epsilon_2 = epsilon * alpha_2
    
    if alpha_1 < alpha or alpha_2 <= 1:
        alpha_1 = alpha + 1
        alpha_2 = alpha_1 + 1
        #return torch.tensor((float('Inf')))

    A = (1-q) / (1 - (q*np.exp(epsilon_2))**((alpha_2-1)/alpha_2))
    B = np.exp(epsilon_1)/(q**(1/(alpha_1 - 1)))

    loss =  1/(alpha-1) * torch.log((1-q) * A**(alpha-1) + q * B**(alpha-1))
    return loss

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

    return logits, indices_to_remove

def epsilon_filtering(logits, e, filter_value=-float("Inf")):
    probs = logits.softmax(dim=-1)
    indicies_to_remove = probs < e
    max_word = torch.argmax(logits, dim=-1)
    indicies_to_remove[..., max_word.squeeze()] = 0
    logits[..., indicies_to_remove] = filter_value
    return logits

def top_k_filtering(logits, top_k, filter_value=-float("Inf")):
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = filter_value
    return logits, indices_to_remove


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ensemble", type=int, default=8)
    parser.add_argument("--model_name", type=str, default="GPT2")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--data_subset", type=str, default="wikitext-103-v1")
    parser.add_argument("--device", type=str, default="cuda:6")
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--query_budget", type=int, default=1024)
    parser.add_argument("--target_multiplier", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=2)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--p_value", type=float, default=1.0)
    parser.add_argument("--e_value", type=float, default=0.01)
    args = parser.parse_args()

    pub_ppl_list = []
    priv_ppl_list = []
    dpsgd_ppl_list = []
    mix_ppl_list = []
    ensemble_ppl_list = []
    for i in tqdm.tqdm(range(0, 5)):
        args.seed = i 
        pub_ppl, priv_ppl, dpsgd_ppl, mix_ppl, ensemble_ppl = main(args)
        pub_ppl_list.append(pub_ppl)
        priv_ppl_list.append(priv_ppl)
        dpsgd_ppl_list.append(dpsgd_ppl)
        mix_ppl_list.append(mix_ppl)
        ensemble_ppl_list.append(ensemble_ppl)

    print(f"Final Perplexity score for Pre-Trained Model: {np.mean(pub_ppl_list):.2f}")
    print(f"Final Perplexity score for Fine-Tuned Model: {np.mean(priv_ppl_list):.2f}")
    print(f"Final Perplexity score for DP-Fine-Tuned Model: {np.mean(dpsgd_ppl_list):.2f}")
    print(f"Final Perplexity score for Mix Model: {np.mean(mix_ppl_list):.2f}")
    print(f"Final Perplexity score for Ensemble Model: {np.mean(ensemble_ppl_list):.2f}")