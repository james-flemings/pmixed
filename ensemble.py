import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from peft import PeftModel, PeftConfig
from typing import List
import torch.nn as nn
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
        self.instance_priv_loss = [0 for _ in range(self.num_ensemble)]
        self.personalized_priv_loss = [[] for _ in range(self.num_ensemble)]
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
            output_dists.append(nn.functional.softmax(logits, dim=1))
        logits = self.pub_model(context).logits.squeeze()
        output_dists.append(nn.functional.softmax(logits, dim=1))
        return output_dists 

    def calc_per_inst_priv_loss(self, p, mixed_dist, lambd, pub_pred):
        N = len(mixed_dist) 
        for i, mix_dist in enumerate(mixed_dist):
            lambd_i = np.mean(self.lambdas[:i] + self.lambdas[i+1:]).item()
            q = (p - mix_dist) * N / (lambd * (N - 1)) * lambd_i
            r = p + (1 - lambd) * pub_pred
            q += (1 - lambd) * pub_pred
            self.instance_priv_loss[i] += max(Ensemble.renyiDiv(r, q, self.alpha),
                                              Ensemble.renyiDiv(q, r, self.alpha)
                                              )

    def calc_indiv_priv_loss(self, mix_dist, pub_pred, i, lambd):
        #return max(torch.log(torch.max(lambd + pub_pred)) -
        #           torch.log(torch.max(lambd * ((N-1)/N + out_dist/N) + pub_pred)),
        #           torch.log(torch.max(lambd/N * out_dist + pub_pred)) -
        #           torch.log(torch.max(pub_pred)) 
        #           )**2
        return max(torch.log(torch.max(mix_dist)) - 
                   torch.log(torch.max((1 - lambd) * pub_pred)), 
                   torch.log(torch.max((1 - lambd) * pub_pred)) -
                   torch.log(torch.max(mix_dist))
                )**2

    def priv_pred(self, output_dists):
        self.lambdas = [self.lambda_solver_bisection(output_dists[i].cpu(),
                                                    output_dists[self.num_ensemble].cpu(),
                                                    i
                                                    ) for i in range(self.num_ensemble)]
        #self.lambdas = [0.99 for _ in range(self.num_ensemble)]
        self.lambda_history.append(np.mean(self.lambdas))
        mixed_dists = [self.mix(output_dists[i], output_dists[self.num_ensemble], self.lambdas[i])
                       for i in range(self.num_ensemble)]

        #mixed_dists = [self.lambdas[i] * output_dists[i] +
        #               (1 - self.lambdas[i]) * output_dist[self.num_ensemble]
        #                for i in range(self.num_ensemble)]

        mixed_dists = torch.stack(mixed_dists)
        ensemble_dist = torch.mean(mixed_dists, dim=0) 
        pub_pred = output_dists[self.num_ensemble]
        #N = len(mixed_dists)
        losses = [self.calc_indiv_priv_loss(mix_dist, pub_pred, i, self.lambdas[i])
                   for i, mix_dist in enumerate(mixed_dists)]
        
        for i, loss in enumerate(losses):
            self.personalized_priv_loss[i].append(loss.cpu())
        
        return ensemble_dist
        '''

        if any(loss > self.target for loss in losses):
            self.lambda_history.append(lambd)
            return (ensemble_dist + pub_pred)

        lambd = 3/4 
        self.lambda_history.append(lambd)

        mixed_dists = torch.stack(output_dists[:-1])
        ensemble_dist = lambd * torch.mean(mixed_dists, dim=0) 
        pub_pred = (1-lambd) * output_dists[self.num_ensemble]
        N = len(mixed_dists)
        losses = [self.calc_indiv_priv_loss(out_dist, pub_pred, N, lambd)
                   for out_dist in output_dists[:-1]]
        
        for i, loss in enumerate(losses):
            self.personalized_priv_loss[i].append(loss.cpu())
        '''

    def lambda_solver_bisection(self, p, pub_pred, i):
        def f(lambd):
            p_mix = self.mix(p, pub_pred, lambd)
            eps = self.calc_indiv_priv_loss(p_mix, pub_pred, i, lambd)
            return (eps - self.target)

        if f(0.99) <= 0.0:
            lambd = 0.99 
        else:
            lambd = bisect(f, 0, 0.99, maxiter=10, disp=False)
        return lambd

    def mix(self, p, p_prime, lambd=0.5):
        return (lambd * p + (1-lambd) * p_prime)


    def reg_pred(self, output_dists):
        output = output_dists[0]
        for i in range(1, self.num_ensemble):
            output += output_dists[i] 
        return output / self.num_ensemble

    def print_priv_losses(self):
        '''
        print("-----------------------------")
        print("Per-instance privacy loss")
        for i, loss in enumerate(self.instance_priv_loss):
            print(f"Model {i}: {loss:.3f}")
        '''
        print("-----------------------------")
        print("Individual privacy loss")
        for i, loss in enumerate(self.personalized_priv_loss):
            print(f"Model {i}: {sum(loss):.3f}")
        print("-----------------------------")
        return
    
    def print_lambdas(self):
        print(f"Average lambda value: {np.mean(self.lambda_history):.2f}")

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