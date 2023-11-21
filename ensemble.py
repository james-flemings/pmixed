import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from typing import List
import torch.nn.functional as F
import numpy as np
from math import comb
import copy
import matplotlib.pyplot as plt
from scipy.optimize import bisect, brentq
import os

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
        self.target = 0
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
    
    def subsample_eps(self, beta, size):
        return 1/(self.alpha-1) * np.log((1-self.p)**(self.alpha-1) * (1 + (self.alpha-1) * self.p) 
                                    + np.sum([comb(self.alpha, k) * (1-self.p)**(self.alpha-k) * self.p**k *
                                              np.exp((k-1) * self.privacy_loss(beta, size, k))
                                                for k in range(2, self.alpha+1)]
                                            )
                                        )
    def privacy_loss(self, beta, size, alpha):
        if size == 1 or size == 0:
            return beta * alpha
        return (np.log((size - 1)/size + np.exp((alpha-1) * 4 * beta * alpha) / size ) 
                / (alpha-1))

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
            output_dists.append(F.softmax(logits, dim=1))
        logits = self.pub_model(context).logits.squeeze().cpu()
        output_dists.append(F.softmax(logits, dim=1))
        return output_dists 

    def poisson_subsample(self, p):
        '''
        Poisson Subsampling is equivalent to m ~ Binomial(n, p), 
        then sampling m without replacement from [N]
        '''
        return np.random.choice(self.num_ensemble, np.random.binomial(self.num_ensemble, p))

    def priv_pred(self, output_dists):
        sampled = self.poisson_subsample(self.p)
        if len(sampled) == 0:
            self.lambda_history.append(0)
            return output_dists[self.num_ensemble]
        self.target = self.beta_solver_bisection(len(sampled)) * self.alpha
        self.lambdas = np.array([self.lambda_solver_bisection(output_dists[i],
                                                    output_dists[self.num_ensemble],
                                                    )for i in sampled])
        self.lambda_history.append(np.mean([lambd for lambd in self.lambdas]))
        mixed_dists = [self.mix(output_dists[i], output_dists[self.num_ensemble], self.lambdas[lamb_i])
                       for lamb_i, i in enumerate(sampled)]
        return torch.stack(mixed_dists).mean(dim=0)

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
    
    def rd_mollification(self, p_mix, p_pub, alpha):
        return  max(self.renyiDiv(p_mix, p_pub, alpha=alpha),
                   self.renyiDiv(p_pub, p_mix, alpha=alpha)).item()

    def lambda_solver_bisection(self, p, p_pub):
        def f(lambd):
            pred = self.mix(p, p_pub, lambd)
            eps = self.rd_mollification(pred, p_pub, self.alpha)
            return (eps - self.target)
        if f(1) <= 0.0:
            lambd = 1 
        else:
            lambd = bisect(f, 0, 1, maxiter=20, disp=False)
        return lambd
    
    def beta_solver_bisection(self, size):
        def f(beta):
            sub_eps = self.subsample_eps(beta, size)
            return (sub_eps - self.eps/self.q_budget)
        return bisect(f, 0, 5, maxiter=20, disp=False)

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

    def plot_lambdas(self):
        x = [i for i in range(len(self.lambda_history))]
        plt.figure().set_figwidth(15)
        plt.plot(x, self.lambda_history)
        plt.savefig(os.path.join("plts", "lambda_vals.png"))
        plt.clf()