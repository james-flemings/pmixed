import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from peft import PeftModel, PeftConfig
from typing import List
import torch.nn as nn
import numpy as np
import copy

class Ensemble():
    def __init__(self, model_dirs, model_name, tokenizer,
                device="cpu", q_budget=1024, alpha=2, eps=2,
                sigma_1 =1, sigma_2=0.5):
        self.device = device
        self.model_name = model_name
        self.q_budget = q_budget
        self.alpha = alpha
        self.eps = eps
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.pub_model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                     pad_token_id=tokenizer.eos_token_id).to(
                                                     self.device)
        self.model_dirs = model_dirs
        self.lora_ensemble = 0 
        self.num_ensemble = len(self.model_dirs)
        self.lambdas = [3/4 for _ in range(self.num_ensemble)]
        self.indiv_eps = [0 for _ in range(self.num_ensemble)]

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
            output_dists.append(nn.functional.softmax(logits, dim=1))
        logits = self.pub_model(context).logits.squeeze()
        output_dists.append(nn.functional.softmax(logits, dim=1))
        return output_dists 

    def calc_privacy_loss(self, dist):
    #    for i, eps in enumerate(self.indiv_eps):
        pass        


    def priv_pred(self, output_dists):
        dim_size = len(output_dists[0])
        ensemble_dist = torch.zeros(dim_size).to(self.device)
        for i in range(self.num_ensemble):
            ensemble_dist += 1/self.num_ensemble * (self.lambdas[i] * 
                                                   output_dists[i]  +
                                                   (1 - self.lambdas[i]) *
                                                   output_dists[self.num_ensemble] 
                                                   )  
        self.calc_privacy_loss(ensemble_dist)
        return ensemble_dist

    def reg_pred(self, output_dists):
        output = output_dists[0]
        for i in range(1, self.num_ensemble):
            output += output_dists[i] 
        return output / self.num_ensemble

    def print_priv_budgets(self):
        for budg in self.indiv_eps:
            print(budg)
        return
    
    def print_lambdas(self):
        for lambd in self.lambdas:
            print(lambd.item())
    