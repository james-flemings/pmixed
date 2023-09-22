import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from peft import PeftModel, PeftConfig
from typing import List
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.optimize import bisect
import os

class Ensemble():
    def __init__(self, model_dirs, model_name, tokenizer,
                device="cpu", q_budget=1024, alpha=2, eps=2, 
                target_mult=1.0, p_value=1.0
    ):
        self.device = device
        self.model_name = model_name
        self.q_budget = q_budget
        self.alpha = alpha
        self.eps = eps
        self.p_value = p_value
        self.pub_model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                     pad_token_id=tokenizer.eos_token_id).to(
                                                     self.device)
        self.model_dirs = model_dirs
        self.num_ensemble = len(self.model_dirs)
        self.lambdas = np.array([1/4 for _ in range(self.num_ensemble)])
        self.priv_loss = []
        self.left_over = 0
        self.target = np.sqrt((2 * self.eps) / (self.q_budget * self.alpha))
        #self.target = (self.eps ) / (self.q_budget )
        self.lambda_history = []

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

    def priv_pred(self, output_dists):
        self.lambdas = [self.lambda_solver_bisection(output_dists[i],
                                                    output_dists[self.num_ensemble],
                                                    ) for i in range(self.num_ensemble)]
        #self.lambdas = [0.1 for i in range(self.num_ensemble)]
        self.lambda_history.append(np.mean([lambd for lambd in self.lambdas]))
        mixed_dists = [self.mix(output_dists[i], output_dists[self.num_ensemble], self.lambdas[i])
                       for i in range(self.num_ensemble)]

        ensemble_dist = torch.stack(mixed_dists).mean(dim=0) 
        pub = output_dists[self.num_ensemble]
        loss = self.calc_dependent_loss(mixed_dists, pub)
        self.left_over += (self.eps/self.q_budget - loss)
        #print("left over", self.left_over)
        #print('loss', loss)
        #print('lambda', self.lambda_history[-1])
        self.priv_loss.append(loss) 
        return ensemble_dist 

    def calc_dependent_loss(self, mixed_dists, p_pub):
        max_loss = 0
        p = torch.stack(mixed_dists).mean(dim=0)
        for i in range(len(mixed_dists)):
            p_i = torch.stack(mixed_dists[:i] + mixed_dists[i+1:]).mean(dim=0)
            eps = max(self.renyiDiv(p, p_i, alpha=self.alpha),
             self.renyiDiv(p_i, p, alpha=self.alpha))
            max_loss = max(eps, max_loss)
        return max_loss
    
    def data_independent_loss(self, p_mix, p_pub):
        return  max(self.renyiDiv(p_mix, p_pub, alpha=self.alpha),
                   self.renyiDiv(p_pub, p_mix, alpha=self.alpha)).item()

    def lambda_solver_bisection(self, p_priv, p_pub):
        def f(lambd):
            pred = lambd * p_priv + (1-lambd) * p_pub
            #eps = self.data_independent_loss(pred, p_pub)
            eps = self.data_independent_loss(pred, p_pub)
            return (eps - (self.eps/self.q_budget + self.left_over))
        if f(1) <= 0.0:
            lambd = 1 
        else:
            lambd = bisect(f, 0, 1, maxiter=20, disp=False)
        return lambd

    def lambda_solver(self, p_priv, p_pub):
        val_1 = ((np.exp((self.target + self.left_over)) - 1) * p_pub) / (p_priv - p_pub)
        val_2 = ((1 / np.exp((self.target + self.left_over)) - 1) * p_pub) / (p_priv - p_pub)
        val = torch.max(val_1, val_2)
        #val = torch.nan_to_num(val, nan=0.0, neginf=0.0)
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

    def plot_individual_loss(self):
        x = [i for i in range(len(self.priv_loss))]
        plt.plot(x, self.priv_loss)
        plt.savefig(os.path.join("plts", "priv_loss.png"))
        plt.clf()

    def plot_lambdas(self):
        x = [i for i in range(len(self.lambda_history))]
        plt.plot(x, self.lambda_history)
        plt.savefig(os.path.join("plts", "lambda_vals.png"))
        plt.clf()