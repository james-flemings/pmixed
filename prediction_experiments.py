#!venv/bin/python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import os 
import argparse
from pmixed import PMixED 
from datasets import load_dataset
from fine_tune_ensemble import group_text_preprocess, pubmed_preprocess, pubmedqa_preprocess
from peft import PeftModel
import copy
import tqdm
import numpy as np
import gc

torch.set_num_threads(1)

def main(args):
    alpha = args.alpha
    epsilon = args.epsilon - np.log((alpha-1)/alpha) + (np.log(args.delta) + np.log(alpha))/(alpha-1)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    pub_model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                    pad_token_id=tokenizer.eos_token_id).to(
                                                    args.device)
    model_dir = os.path.join("models", f"{args.num_ensemble}_ensemble")
    model_paths = None
    dataset_name = args.dataset if args.subset == None else args.subset
    #dataset_name = "wikitext-103-raw-v1"
    model_paths = [os.path.join(model_dir, f"lora-{args.model_name}-{i}-finetuned-{dataset_name}")
                for i in range(args.num_ensemble)]

    pub_model.eval()
    priv_ensemble = PMixED(pub_model,
                           model_paths,
                           args.model_name,
                           tokenizer,
                           args.device,
                           q_budget=args.query_budget,
                           alpha=args.alpha,
                           delta=args.delta,
                           p=args.p,
                           eps=epsilon,
                           beta=args.beta,
                           lambd=args.lambd,
                           threshold=args.threshold,
                           screen_top_k=args.screen_top_k,
                           sigma=args.sigma,
                           accounting_method=args.accounting_method
    )
    #fine_tuned_model_dir = os.path.join("models", f"lora-{args.model_name}-finetuned-{dataset_name}")
    '''
    fine_tuned_model_dir = os.path.join("models", f"lora-{args.model_name}-finetuned-wikitext-103-raw-v1")
    fine_tuned_model = PeftModel.from_pretrained(copy.deepcopy(pub_model),
                                                 fine_tuned_model_dir,
                                                 pad_token_id=tokenizer.eos_token_id).to(
                                                 args.device)
    '''
    dp_fine_tuned_model = 0
    dp_fine_tuned_model = torch.load(os.path.join("models", f"lora-{args.model_name}-8.0-dp-finetuned-{dataset_name}.pt")).to(args.device)
    #dp_fine_tuned_model = torch.load(os.path.join("models", f"lora-{args.model_name}-8.0-dp-finetuned-wikitext.pt")).to(args.device)
    fine_tuned_model = dp_fine_tuned_model 

    dataset_name = "ccdv/mediasum" if args.dataset == "mediasum" else args.dataset
    if dataset_name == "qiaojin/PubMedQA":
        args.subset = "pqa_unlabeled"
    dataset = load_dataset(dataset_name, args.subset)

    if args.dataset == "qiaojin/PubMedQA":
        dataset = pubmedqa_preprocess(dataset['train'], 'test')

    preprocess_function = group_text_preprocess
    header = 'document' if args.dataset == "mediasum" else 'text'

    test_data = dataset['test'].map(preprocess_function,
                                    fn_kwargs={"tokenizer": tokenizer,
                                               "block_size": args.seq_length,
                                               "header": header},
                                    batched=True,
                                    num_proc=4,
                                    desc="tokenizing dataset",
                                    remove_columns=dataset['test'].column_names
                                    )
    test_data.set_format(type="torch")

    pub_neg_log_likelihood = []
    fine_tuned_neg_log_likelihood = []
    dp_fine_tuned_neg_log_likelihood = []
    ensemble_neg_log_likelihood= []
    step_size = args.query_budget // args.seq_length
    test_loader = DataLoader(test_data.select([i + args.start for i in range(args.start+step_size)]))#, shuffle=True)
    fine_tuned_model.eval()
    k = 0
    for i, data in tqdm.tqdm(enumerate(test_loader), desc="Iterating through test data", total=len(test_loader)):
        labels = data['labels'].to(args.device)
        input_ids = data['input_ids'].to(args.device)
        with torch.no_grad():
            pub_output_logits = pub_model(input_ids).logits 
            fine_tuned_output_logits =fine_tuned_model(input_ids).logits 
            dp_fine_tuned_output_logits = dp_fine_tuned_model(input_ids).logits 
            pub_dist, priv_dists = priv_ensemble.gen_output_dist(input_ids)
            ensemble_logits = []

            if k < args.query_budget:
                for j in tqdm.tqdm(range(args.seq_length), desc="Mixing"):
                    #priv_dists_token = [priv_dist[j].to(args.device) for priv_dist in priv_dists]
                    priv_dists_token = torch.stack([priv_dist[j] for priv_dist in priv_dists])
                    pub_dist_token = pub_dist[j]#.to(args.device)
                    ensemble_output_dist = priv_ensemble.gen_priv_output_dist(pub_dist_token.to(args.device),
                                                                               priv_dists_token.to(args.device))
                    ensemble_logits.append(torch.log(ensemble_output_dist.cpu()))
                    k += 1
                    #priv_dists_token = [priv_dist.cpu() for priv_dist in priv_dists_token]
                    del ensemble_output_dist, priv_dists_token, pub_dist_token
                    #torch.cuda.empty_cache()
                    #gc.collect()
                    #print("Smooth Sensitivity", priv_ensemble.ss)

                ensemble_logits = torch.stack(ensemble_logits)
                ensemble_neg_log_likelihood.append(calc_loss(ensemble_logits, labels.cpu()))
                del ensemble_logits, priv_dists, pub_dist
                torch.cuda.empty_cache()
            else:
                priv_loss = PMixED.convert_to_aprox_dp(priv_loss=priv_ensemble.priv_loss,
                                                       delta=args.delta,
                                                       alpha=args.alpha)
                print("Query budget exhausted")
                print(f'Query budget: {k}; Privacy Loss: ε={priv_loss:.3f}\n\n')
                break
            priv_loss = PMixED.convert_to_aprox_dp(priv_loss=priv_ensemble.priv_loss,
                                                    delta=args.delta,
                                                    alpha=args.alpha)
            print(f'Query budget: {k}; Privacy Loss: ε={priv_loss:.3f}\n\n')

        pub_neg_log_likelihood.append(calc_loss((pub_output_logits), labels))
        fine_tuned_neg_log_likelihood.append(calc_loss((fine_tuned_output_logits), labels))
        dp_fine_tuned_neg_log_likelihood.append(calc_loss(dp_fine_tuned_output_logits, labels))

        del pub_output_logits, fine_tuned_output_logits, dp_fine_tuned_output_logits

    pre_trained_ppl = torch.exp(torch.stack(pub_neg_log_likelihood))
    fine_tuned_ppl = torch.exp(torch.stack(fine_tuned_neg_log_likelihood))
    dp_fine_tuned_ppl = torch.exp(torch.stack(dp_fine_tuned_neg_log_likelihood))
    ensemble_ppl = torch.exp(torch.stack(ensemble_neg_log_likelihood))

    priv_ensemble.print_lambdas()
    priv_ensemble.plot_lambdas()
    if args.threshold is not None:
        priv_ensemble.print_noisy_rd()

    return pre_trained_ppl.mean().cpu(), \
        fine_tuned_ppl.mean().cpu(), \
        dp_fine_tuned_ppl.mean().cpu(), \
        ensemble_ppl.mean().cpu(), \
        priv_ensemble.priv_loss, \
        priv_ensemble.num_noisy, \
        priv_ensemble.num_non_sample, \
        priv_ensemble.num_noise

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
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:6")
    parser.add_argument("--accounting_method", type=str, default=None)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--query_budget", type=int, default=1024)
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--alpha", type=int, default=3)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--lambd", type=float, default=None)
    parser.add_argument("--beta", type=float, default=0.09)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--screen_top_k", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    pub_ppl_list = []
    ft_ppl_list = []
    dpsgd_ppl_list = []
    ensemble_ppl_list = []
    eps_list = []
    num_noisy_list = []
    num_noise_list = []
    step_size = args.query_budget // args.seq_length
    for i in tqdm.tqdm(range(0, args.iters), desc="Runs"):
        #args.start = i * step_size 
        pub_ppl, ft_ppl, dpsgd_ppl, ensemble_ppl, priv_loss, num_noisy, num_non_sample, num_noise = main(args)
        pub_ppl_list.append(pub_ppl)
        ft_ppl_list.append(ft_ppl)
        dpsgd_ppl_list.append(dpsgd_ppl)
        ensemble_ppl_list.append(ensemble_ppl)
        eps = priv_loss + np.log((args.alpha-1)/args.alpha) - (np.log(args.delta) + np.log(args.alpha))/(args.alpha-1)
        eps_list.append(eps)
        num_noisy_list.append(num_noisy)

        print(f"Total privacy loss of PMixED: {eps:.3f}")
        print(f"Number of times used Noisy Mechanism PMixED: {num_noisy}")
        print(f"Average total noise added from Noisy Mech: {np.mean(num_noise)}")
        print(f"Number of times no model was sampled PMixED: {num_non_sample}")

    print(f"Perplexity score of public model: {np.mean(pub_ppl_list):.2f}")
    print(f"Perplexity score of fine-tuned model: {np.mean(ft_ppl_list):.2f}")
    print(f"Perplexity score of DP-SGD: {np.mean(dpsgd_ppl_list):.2f}")
    print(f"Perplexity score of PMixED: {np.mean(ensemble_ppl_list):.2f}")

    print(f"STD Perplexity score of public model: {np.std(pub_ppl_list):.2f}")
    print(f"STD Perplexity score of fine-tuned model: {np.std(ft_ppl_list):.2f}")
    print(f"STD Perplexity score of DP-SGD: {np.std(dpsgd_ppl_list):.2f}")
    print(f"STD Perplexity score of PMixED: {np.std(ensemble_ppl_list):.2f}")

    print(f"Average Privacy loss of PMixED: {np.mean(eps_list):.3f}")
    print(f"Average number of times threshold not met PMixED: {np.mean(num_noisy_list):.2f}")
