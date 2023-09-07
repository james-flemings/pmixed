import argparse
import os
import tqdm
import copy

import numpy as np
from scipy.optimize import bisect

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from peft import PeftModel


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=2.0)
    parser.add_argument("--query_budget", type=int, default=512)
    parser.add_argument("--model_name", type=str, default="GPT2")
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--T", type=int, default=512)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--data_subset", type=str, default="wikitext-103-v1")
    parser.add_argument("--start_context", type=int, default=32)
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    pub_model = GPT2LMHeadModel.from_pretrained(args.model_name)
    pub_model.to(args.device)
    
    fine_tuned_model_dir = os.path.join("models", f"lora-{args.model_name}-finetuned-{args.data_subset}")
    priv_model = PeftModel.from_pretrained(copy.deepcopy(pub_model),
                                                 fine_tuned_model_dir,
                                                 pad_token_id=tokenizer.eos_token_id).to(
                                                 args.device)

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
        fn_kwargs={"block_size": args.seq_length},
        batched=True,
        num_proc=4
    ) 

    test_data = lm_dataset['test']
    test_data.set_format(type="torch")
    test_loader = DataLoader(test_data)#, shuffle=True)

    pub_model.eval()
    priv_model.eval()

    pub_nll = []
    priv_nll = []
    ground_nll = []
    mix_nll = []
    priv_loss = []
    lambdas = []

    c = args.start_context
    target = args.epsilon / args.query_budget
    for i, data in enumerate(tqdm.tqdm(test_loader)):
        input_ids = data['input_ids'].to(args.device)
        labels = data['labels'].to(args.device)
        pub_input_ids = input_ids[: , :c]
        priv_input_ids = input_ids[: , :c]
        mix_input_ids = input_ids[: , :c]
        with torch.no_grad():
            for _ in tqdm.tqdm(range(args.seq_length-c)):
                pub_logits = pub_model(pub_input_ids).logits.squeeze()[-1:, :]
                priv_logits = priv_model(priv_input_ids).logits.squeeze()[-1:, :]
                mix_pub_logits = pub_model(mix_input_ids).logits.squeeze()[-1:, :]
                mix_priv_logits = priv_model(mix_input_ids).logits.squeeze()[-1:, :]

                pub_logits_filtered, _ = top_p_filtering(pub_logits.clone(), args.p)
                priv_logits_filtered, _ = top_p_filtering(priv_logits.clone(), args.p)
                #mix_pub_logits_filtered, ind = top_p_filtering(mix_pub_logits.clone(), args.p)
                mix_priv_logits_filtered, ind = top_p_filtering(mix_priv_logits.clone(), args.p)
                mix_pub_logits_filtered = mix_pub_logits.clone()
                mix_pub_logits_filtered[ind] = -float('Inf')
                #mix_priv_logits_filtered = mix_priv_logits.clone()
                #mix_priv_logits_filtered[ind] = -float('Inf')
                
                pub_probs = F.softmax(pub_logits_filtered, dim=-1)
                priv_probs = F.softmax(priv_logits_filtered, dim=-1)
                mix_pub_probs = F.softmax(mix_pub_logits_filtered, dim=-1)
                mix_priv_probs = F.softmax(mix_priv_logits_filtered, dim=-1)

                mix_probs, loss, lambd = private_pred(mix_priv_probs[-1, :], mix_pub_probs[-1, :], target)
                priv_loss.append(loss)
                lambdas.append(lambd)

                pub_next_indicies = pub_probs.multinomial(1).view(1, -1)
                priv_next_indicies = priv_probs.multinomial(1).view(1, -1)
                mix_next_indicies = mix_probs.multinomial(1).view(1, -1)

                pub_input_ids = torch.cat([pub_input_ids, pub_next_indicies[:, :1]], dim=1)
                priv_input_ids = torch.cat([priv_input_ids, priv_next_indicies[:, :1]], dim=1)
                mix_input_ids = torch.cat([mix_input_ids, mix_next_indicies[:, :1]], dim=1)

            ground_nll.append(calc_loss(pub_model(input_ids).logits.squeeze()[c:, :], labels[:, c:]))
            pub_nll.append(calc_loss(pub_model(pub_input_ids).logits.squeeze()[c:, :],
                                     pub_input_ids[:, c:]))
            priv_nll.append(calc_loss(pub_model(priv_input_ids).logits.squeeze()[c:, :],
                                      priv_input_ids[:, c:]))
            mix_nll.append(calc_loss(pub_model(mix_input_ids).logits.squeeze()[c:, :],
                                      mix_input_ids[:, c:]))
            #for j in tqdm.tqdm(range(args.seq_length-args.start_context)):
            break

        #if i < args.query_budget // args.seq_length:
    ground_ppl = torch.exp(torch.stack(ground_nll))
    pub_ppl = torch.abs(torch.exp(torch.stack(pub_nll)) - ground_ppl)
    priv_ppl = torch.abs(torch.exp(torch.stack(priv_nll)) - ground_ppl)
    mix_ppl = torch.abs(torch.exp(torch.stack(mix_nll)) - ground_ppl)

    print(f"Perplexity score for Ground Truth: {ground_ppl.mean():.2f}")
    print(f"Perplexity score for Pre-Trained Model: {pub_ppl.mean():.2f}")
    print(f"Perplexity score for Fine-Tuned Model: {priv_ppl.mean():.2f}")
    print(f"Perplexity score for Mix Model: {mix_ppl.mean():.2f}")

    print(f"Total privacy loss of model: {sum(priv_loss):.4f}")
    print(f"Max loss", max(priv_loss))
    print(f"Average lambda value: {np.mean(lambdas):.4f}")
    print(f"Min lambda: {np.min(lambdas)}")
    print(f"Max lambda: {np.max(lambdas)}")

def calc_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = nn.CrossEntropyLoss()
    return loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

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

def private_pred(priv_model, pub_model, target, alpha=2):
    lambd = lambda_solver_bisection(priv_model.cpu(), pub_model.cpu(), target, 2*alpha)
    #lambd = 0.4
    pred = lambd * priv_model + (1-lambd) * pub_model
    loss = max(renyiDiv(pred.cpu(), pub_model.cpu(), alpha=2*alpha),
               renyiDiv(pub_model.cpu(), pred.cpu(), alpha=2*alpha)).item()
    return pred, loss, lambd

def lambda_solver_bisection(p_priv, p_pub, target, alpha):
    def f(lambd):
        pred = lambd * p_priv + (1-lambd) * p_pub
        eps = max(renyiDiv(pred, p_pub, alpha=alpha), renyiDiv(p_pub, pred, alpha=alpha))
        return (eps - target/2)

    if f(1) <= 0.0:
        lambd = 1 
    else:
        lambd = bisect(f, 0, 1, maxiter=10, disp=False)
    return lambd

def calc_priv_loss(p_pub, pred):
    ind = torch.nonzero(p_pub)
    loss = torch.max(torch.max(p_pub[ind]/pred[ind]), torch.max(pred[ind]/p_pub[ind]))
    return (torch.log(loss)**2) / 2

def renyiDiv(p, q, alpha=float('inf')):
    if alpha == float('inf'):
        RD = torch.log(torch.max(p/q))
    elif alpha == 1:
        RD = torch.sum(p*torch.log(p/q))
    else:
        inds = torch.nonzero(q)
        #r = (p[inds]**alpha)/(q[inds]**(alpha-1))
        #RD = 1/(alpha-1)*torch.log(torch.sum(r))
        RD = 1/(alpha-1)*torch.log(
            torch.sum((p[inds]**alpha)/(q[inds]**(alpha-1))))
    if torch.isnan(RD):
        RD = torch.log(torch.max(p/q))
    return RD 

if __name__ == "__main__":
    main()
