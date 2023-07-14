import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from peft import PeftModel, PeftConfig
from typing import List
import torch.nn as nn
import numpy as np

class Ensemble():
    def __init__(self, model_dirs, pub_model, 
                device="cpu", q_budget=1024, alpha=2, eps=2,
                sigma_1 =1, sigma_2=0.5):
        self.device = device
        self.q_budget = q_budget
        self.alpha = alpha
        self.eps = eps
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.pub_model = pub_model
        self.model_dirs = model_dirs
        self.lora_ensemble = 0 
        self.num_ensemble = len(self.model_dirs)
        self.lambdas = [1 for _ in range(self.num_ensemble)]
        self.indiv_eps = [0 for _ in range(self.num_ensemble)]

        for i, dir in enumerate(self.model_dirs):
            if i == 0:
                self.lora_ensemble = PeftModel.from_pretrained(pub_model, 
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
            output_dists.append(nn.functional.softmax(logits, dim=1).to('cpu'))
        logits = self.pub_model(context).logits.squeeze()
        output_dists.append(nn.functional.softmax(logits, dim=1).to('cpu'))
        return output_dists 

    def calc_privacy_loss(self, pred, i):
        #self.indiv_eps[i] += self.alpha * (self.lambdas[i] / 
        #                                   self.num_ensemble * pred)**2 / (2 * 
        #                                   self.simga_1**2)
        return self.alpha * (self.lambdas[i]**2 / 
                                self.num_ensemble**2 * torch.sum(pred**2)) / (2 * 
                                self.sigma_2**2) 


    def priv_pred(self, output_dists):
        cur_eps = []
        for i in range(self.num_ensemble):
            cur_eps.append(self.calc_privacy_loss(output_dists[i], i))
            if self.indiv_eps[i] > self.eps:
                self.lambdas[i] = 0
        dim_size = len(output_dists[0])
        ensemble_dist = torch.zeros(dim_size) 
        for i in range(self.num_ensemble):
            ensemble_dist += 1/self.num_ensemble * (self.lambdas[i] * 
                                                   output_dists[i] +
                                                   (1 - self.lambdas[i]) *
                                                   output_dists[self.num_ensemble]
                                                   )
        ensemble_dist +=  np.random.normal(0, self.sigma_2, dim_size)
        # This still gets negative numbers
        ensemble_dist += 2 * torch.min(ensemble_dist) 
        ensemble_dist /= torch.sum(ensemble_dist) 

        for i in range(self.num_ensemble):
            self.indiv_eps[i] += cur_eps[i] if self.lambdas[i] != 0 else 0 

        for i in range(self.num_ensemble):
            self.lambdas[i] = self.sigma_2 * self.num_ensemble / np.sqrt(
                                                torch.sum(output_dists[i]**2)) * np.sqrt(
                                                            (2 * self.eps) /
                                                            (self.alpha * self.q_budget)
                                                            )
        
        return ensemble_dist

    def reg_pred(self, output_dists):
        ensemble_dist = torch.zeros(len(output_dists[0]))
        for i in range(self.num_ensemble):
            ensemble_dist += output_dists[i] / self.num_ensemble
        return ensemble_dist 

    def print_priv_budgets(self):
        for budg in self.indiv_eps:
            print(budg)