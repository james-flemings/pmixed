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
                target_mult=1.0
    ):
        self.device = device
        self.model_name = model_name
        self.q_budget = q_budget
        self.alpha = alpha
        self.eps = eps
        self.pub_model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                     pad_token_id=tokenizer.eos_token_id).to(
                                                     self.device)
        self.model_dirs = model_dirs
        self.num_ensemble = len(self.model_dirs)
        self.lambdas = np.array([1/4 for _ in range(self.num_ensemble)])
        self.personalized_priv_loss = [[] for _ in range(self.num_ensemble)]
        self.left_over_priv_loss = [0 for _ in range(self.num_ensemble)]
        self.pairings = [(i, i+1) for i in range(0, self.num_ensemble, 2)]
        self.target = self.eps / self.q_budget
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
            logits = self.lora_ensemble(context).logits.squeeze()
            output_dists.append(F.softmax(logits))
        logits = self.pub_model(context).logits.squeeze()
        #logits_filtered = self.top_p_filtering(logits.clone(), 0.95)
        #output_dists.append(F.softmax(logits_filtered))
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

    def calc_priv_loss(self, pred, p_pub):
        #ind = torch.nonzero(p_pub)
        #loss = torch.max(torch.max(p_pub[ind]/pred[ind]), torch.max(pred[ind]/p_pub[ind]))
        loss = torch.max(torch.max(p_pub/pred), torch.max(pred/p_pub))
        return (torch.log(loss)**2) / 2

    def priv_pred(self, output_dists):
        self.lambdas = [self.lambda_solver_bisection(output_dists[i].cpu(),
                                                    output_dists[self.num_ensemble].cpu(),
                                                    i
                                                    ) for i in range(self.num_ensemble)]
        self.lambda_history.append(np.mean(self.lambdas))
        mixed_dists = [self.mix(output_dists[i], output_dists[self.num_ensemble], self.lambdas[i])
                       for i in range(self.num_ensemble)]

        mixed_dists = torch.stack(mixed_dists)
        ensemble_dist = torch.mean(mixed_dists, dim=0) 
        pub_pred = output_dists[self.num_ensemble]
        losses = [self.renyiDiv(mix_dist.cpu(), pub_pred.cpu(), alpha=self.alpha) for mix_dist in mixed_dists]
        
        for i, loss in enumerate(losses):
            self.personalized_priv_loss[i].append(loss.cpu())
            #self.left_over_priv_loss[i] += (self.target/2 - loss.cpu().item())
        
        return ensemble_dist

    def lambda_solver_bisection(self, p_priv, p_pub, i):
        def f(lambd):
            pred = lambd * p_priv + (1-lambd) * p_pub
            #eps = self.calc_priv_loss(pred, p_pub)
            #return (eps - (self.target/2 + self.left_over_priv_loss[i]))
            eps = self.renyiDiv(pred, p_pub, alpha=self.alpha)
            return (eps - self.target/2)
        if f(1) <= 0.0:
            lambd = 1 
        else:
            lambd = bisect(f, 0, 1, maxiter=20, disp=False)
        return lambd

    def mix(self, p, p_prime, lambd=0.5):
        return (lambd * p + (1-lambd) * p_prime)

    def reg_pred(self, output_dists):
        output = output_dists[0]
        for i in range(1, self.num_ensemble):
            output += output_dists[i] 
        return output / self.num_ensemble

    def print_priv_losses(self):
        print("-----------------------------")
        print("Individual privacy loss")
        for i, loss in enumerate(self.personalized_priv_loss):
            print(f"Model {i}: {sum(loss):.3f}")
        print("-----------------------------")
        return
    
    def print_lambdas(self):
        print(f"Average lambda value: {np.mean(self.lambda_history):.3f}")

    def plot_individual_loss(self):
        x = [i for i in range(len(self.personalized_priv_loss[0]))]
        plt.plot(x, self.personalized_priv_loss[0])
        plt.savefig(os.path.join("plts", "priv_loss.png"))
        plt.clf()

    def plot_lambdas(self):
        x = [i for i in range(len(self.lambda_history))]
        plt.plot(x, self.lambda_history)
        plt.savefig(os.path.join("plts", "lambda_vals.png"))
        plt.clf()