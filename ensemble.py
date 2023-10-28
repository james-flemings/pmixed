import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from peft import PeftModel, PeftConfig
from typing import List
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import comb
import copy
import matplotlib.pyplot as plt
from scipy.optimize import bisect, brentq
import os
from itertools import combinations
import tqdm

class Ensemble():
    def __init__(self, model_dirs, model_name, tokenizer,
                device="cpu", q_budget=1024, alpha=2, eps=2, delta=1e-5,
                p=1.0
    ):
        self.device = device
        self.model_name = model_name
        self.q_budget = q_budget
        self.alpha = alpha 
        self.eps = eps
        self.p = p
        self.delta = delta
        self.pub_model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                     pad_token_id=tokenizer.eos_token_id).to(
                                                     self.device)
        self.model_dirs = model_dirs
        self.num_ensemble = len(self.model_dirs)
        self.lambdas = np.array([0 for _ in range(self.num_ensemble)])
        self.lambda_history = []
        self.priv_loss = [] 

        if p < 1.0:
            a = 1 
            while self.subsample_eps(a * self.eps / self.q_budget, self.p, self.alpha) < self.eps / self.q_budget:
                a += 1
            print("a value", (a-1))
            self.eps = self.eps * (a-1)

        self.target = np.log(self.num_ensemble * np.exp((self.alpha-1) * alpha * self.eps / self.q_budget)
                        + 1 - self.num_ensemble) / (4*(self.alpha - 1))
        print(f"Target value {self.target:2f}")
        self.lora_ensemble = 0 
        for i, dir in enumerate(self.model_dirs):
            if i == 0:
                self.lora_ensemble = PeftModel.from_pretrained(copy.deepcopy(
                                                             self.pub_model), 
                                                             dir,
                                                             adapter_name=f"lora-{i}"
                                                            ).to(self.device)
            else:
                self.lora_ensemble.load_adapter(dir, adapter_name=f"lora-{i}")
    
    def subsample_eps(self, eps, p, alpha):
        return 1/(alpha-1) * np.log((1-p)**(alpha-1) * (1+ (alpha-1) * p) 
                                    + np.sum([comb(alpha, k) * (1-p)**(alpha-k) * p**k *
                                              np.exp((k-1) * eps * k) for k in range(2, alpha+1)]))

    @staticmethod
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

    def pred_dist(self, context):
        output_dists = []
        for i in range(self.num_ensemble):
            self.lora_ensemble.set_adapter(f"lora-{i}")
            logits = self.lora_ensemble(context).logits.squeeze().cpu()
            output_dists.append(F.softmax(logits))
        logits = self.pub_model(context).logits.squeeze().cpu()
        output_dists.append(F.softmax(logits))
        return output_dists 

    def top_p_filtering(self, logits, p, filter_value=-float("Inf")):
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
    
    def subsample(self, candidates, p):
        sampled = []
        #for c in candidates:
        #    if np.random.uniform() < p:
        #        sampled.append(c)
        sampled = np.random.choice(self.num_ensemble, int(p*self.num_ensemble))
        return sampled

    def priv_pred(self, output_dists):
        sampled = self.subsample([i for i in range(self.num_ensemble)], self.p)
        loss = self.subsample_eps(self.eps/self.q_budget, self.p, self.alpha)
        self.priv_loss.append(loss) 
        #if len(sampled) == 0:
        #    self.lambda_history.append(0)
        #    return output_dists[self.num_ensemble]

        self.lambdas = np.array([self.lambda_solver_bisection(output_dists[i],
                                                    output_dists[self.num_ensemble],
                                                    )for i in sampled])

        self.lambda_history.append(np.mean([lambd for lambd in self.lambdas]))
        mixed_dists = [self.mix(output_dists[i], output_dists[self.num_ensemble], self.lambdas[lamb_i])
                       for lamb_i, i in enumerate(sampled)]

        mixed_dists = torch.stack(mixed_dists)
        ensemble_dist = mixed_dists.mean(dim=0) 
        return ensemble_dist

    def data_dependent_loss(self, mixed_dists, alpha, ret_idx=False):
        max_loss = 0
        idx = -1 
        p = mixed_dists.mean(dim=0)
        for i in range(mixed_dists.size()[0]):
            p_i = torch.cat((mixed_dists[:i, :], mixed_dists[i+1:, :])).mean(dim=0)
            eps = max(self.renyiDiv(p, p_i, alpha=alpha),
             self.renyiDiv(p_i, p, alpha=alpha))
            max_loss = max(max_loss, eps)
        return (max_loss, idx.pop()) if ret_idx else max_loss
    
    def data_dependent_loss_ub(self, mixed_dists, pub, beta, sigma):
        ss = self.smooth_sensitivity(mixed_dists, pub, beta)
        eps = self.data_dependent_loss(mixed_dists, alpha=self.alpha)
        b = beta * np.random.normal(loc=0.0, scale=0.5*sigma, size=1)
        c = np.sqrt(2 * np.log(2/(self.delta/2))) 
        mu = np.log(ss) + b + (c * 0.5*sigma * beta)
        return (eps + ss * np.random.normal(loc=0.0, scale=sigma, size=1)
                 + self.sigma * c * np.exp(mu))

    def smooth_sensitivity(self, mixed_dists, p_pub, beta, k=2):
        X = [i for i in range(self.num_ensemble)] 
        ss = 0
        used_idx = [] 
        eps = 0
        for i in range(self.num_ensemble):
            for j in range(i+1, self.num_ensemble):
                p = torch.cat((mixed_dists[[i]], mixed_dists[[j]]))
                eps_prime = self.data_dependent_loss(p, alpha=float('Inf'))
                if  eps_prime > eps:
                    used_idx = [i, j]
                    eps = eps_prime 
        ranked_dists = torch.cat((mixed_dists[[used_idx[0]]], mixed_dists[[used_idx[1]]]))

        full_ids = {i for i in range(self.num_ensemble)}
        for _ in range(self.num_ensemble-2):
            ls = 0
            id = 0
            eps = self.data_dependent_loss(ranked_dists, alpha=float("Inf"))
            for m in full_ids.difference(set(used_idx)):
                eps_prime = self.data_dependent_loss(torch.cat((ranked_dists, mixed_dists[[m]])), alpha=float("Inf"))
                ls_prime = abs(eps - eps_prime)
                if ls_prime > ls:
                    ls = ls_prime
                    id = m
            used_idx.append(id)
            ranked_dists = torch.cat((ranked_dists, mixed_dists[[id]]))

        ss_prime_eff = np.exp(-beta * k) * self.local_sensitivity(ranked_dists[:3, :]) 
        return ss_prime_eff

    def local_sensitivity(self, mixed_dists):
        eps = self.data_dependent_loss(mixed_dists, alpha=self.alpha)
        X = combinations([i for i in range(len(mixed_dists))], len(mixed_dists)-1)
        return max([abs(eps - self.data_dependent_loss(mixed_dists[list(x)], alpha=self.alpha, ret_idx=False)) for x in X])
    
    def data_independent_loss(self, p_mix, p_pub, alpha):
        return  max(self.renyiDiv(p_mix, p_pub, alpha=alpha),
                   self.renyiDiv(p_pub, p_mix, alpha=alpha)).item()

    def lambda_solver_bisection(self, p, p_pub):
        def f(lambd):
            pred = self.mix(p, p_pub, lambd)
            eps = self.data_independent_loss(pred, p_pub, self.alpha)
            return (eps - self.target)
        if f(1) <= 0.0:
            lambd = 1 
        else:
            lambd = bisect(f, 0, 1, maxiter=100, disp=False)
        return lambd
    
    def lambda_solver_iter(self, p, p_pub, lambd):
        while lambd > 0:
            pred = self.mix(p, p_pub, lambd)
            eps = self.data_independent_loss(pred, p_pub, self.alpha)
            if eps < self.target:
                return lambd
            lambd -= 0.01
        return lambd


    def lambda_solver(self, p_priv, p_pub):
        val_1 = ((np.exp(self.target) - 1) * p_pub) / (p_priv - p_pub)
        val_2 = ((1 / np.exp(self.target) - 1) * p_pub) / (p_priv - p_pub)
        val = torch.max(val_1, val_2)
        val = torch.min(val, torch.ones(val.size()[0]))
        return val

    def mix(self, p, p_prime, lambd=0.5):
        return (lambd * p + (1-lambd) * p_prime)

    def print_priv_losses(self):
        print("-----------------------------")
        print(f"Total privacy loss {sum(self.priv_loss):.3f}")
        print("-----------------------------")
    
    def print_lambdas(self):
        print("-----------------------------")
        print(f"Average lambda value: {np.mean(self.lambda_history):.3f}")
        print("-----------------------------")
        print(f"Min lambda value: {np.min(self.lambda_history)}")
        print("-----------------------------")
        print(f"Max lambda value: {np.max(self.lambda_history):.3f}")
        print("-----------------------------")

    def plot_individual_loss(self):
        x = [i for i in range(len(self.priv_loss))]
        plt.plot(x, self.priv_loss)
        plt.savefig(os.path.join("plts", "priv_loss.png"))
        plt.clf()

    def plot_lambdas(self):
        x = [i for i in range(len(self.lambda_history))]
        plt.figure().set_figwidth(15)
        plt.plot(x, self.lambda_history)
        plt.savefig(os.path.join("plts", "lambda_vals.png"))
        plt.clf()