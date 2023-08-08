import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from peft import PeftModel, PeftConfig
from typing import List
import torch.nn as nn
import numpy as np
import copy
import matplotlib.pyplot as plt

class Ensemble():
    def __init__(self, model_dirs, model_name, tokenizer,
                device="cpu", q_budget=1024, alpha=2, eps=2
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
        self.lora_ensemble = 0 
        self.num_ensemble = len(self.model_dirs)
        self.lambdas = np.array([1/4 for _ in range(self.num_ensemble)])
        self.instance_priv_loss = [0 for _ in range(self.num_ensemble)]
        self.personalized_priv_loss = [[] for _ in range(self.num_ensemble)]

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
            #output_dists.append(logits)
        logits = self.pub_model(context).logits.squeeze()
        output_dists.append(nn.functional.softmax(logits, dim=1))
        #output_dists.append(logits)
        return output_dists 

    def calc_per_inst_priv_loss(self, p, mixed_dist, pub_pred):
        N = self.num_ensemble
        for i in range(self.num_ensemble):
            q = (p - mixed_dist[i]) * N / (N - 1)
            r = p + pub_pred
            q += pub_pred
            self.instance_priv_loss[i] += max(Ensemble.renyiDiv(r, q, self.alpha),
                                              Ensemble.renyiDiv(q, r, self.alpha)
                                              )



    def calc_indiv_priv_loss(self, p, mixed_dist, lambd, pub_pred, active_set):
        losses = []
        N = len(active_set)
        for i in active_set:
            val, ind = torch.max(mixed_dist[i], 0)
            '''
            loss = (torch.log(1 + lambd * output[i][ind] /
                                                        (self.num_ensemble *
                                                        pub_pred[ind])
                                                        )
                                              )
            if loss > 1:
                print(f"Model {i}: {loss:.2f}")
            self.personalized_priv_loss[i] += loss
            '''
            r = mixed_dist[i] + pub_pred
            q = copy.deepcopy(pub_pred)
            r[ind] += (N-1) / N * lambd 
            q[ind] += lambd
            loss = max(Ensemble.renyiDiv(r, q, self.alpha).cpu(),
                       Ensemble.renyiDiv(q, r, self.alpha).cpu()
            )
            self.personalized_priv_loss[i].append(loss)
            losses.append(loss)
        return losses


    def priv_pred(self, output_dists):
        self.lambdas = np.array([0.05 for _ in range(self.num_ensemble)])
        lambd = np.mean(self.lambdas).item()
        mixed_dists = [lambd * output_dists[i] / self.num_ensemble for i in range(self.num_ensemble)]
        ensemble_dist = sum(mixed_dists) 
        pub_pred = (1 - lambd) * output_dists[self.num_ensemble]
        ensemble_dist += pub_pred 
        active_set = [i for i in range(self.num_ensemble)]
        self.calc_per_inst_priv_loss(ensemble_dist-pub_pred, mixed_dists, pub_pred)
        losses = self.calc_indiv_priv_loss(ensemble_dist-pub_pred, mixed_dists,
                                         lambd, pub_pred, active_set)

        active_set = [i for i, loss in enumerate(losses) if loss < self.eps / (2 * self.q_budget)]
        if len(active_set) <= self.num_ensemble/2:
            return output_dists[self.num_ensemble]
        self.lambdas = np.array([0.5 for _ in range(self.num_ensemble)])
        lambd = np.mean(self.lambdas).item()
        mixed_dists = [lambd * output_dists[i] / len(active_set) if i in active_set else 0 for i in range(self.num_ensemble)] 
        ensemble_dist = sum(mixed_dists)
        pub_pred = (1 - lambd) * output_dists[self.num_ensemble]
        ensemble_dist += pub_pred
        losses = self.calc_indiv_priv_loss(ensemble_dist-pub_pred, mixed_dists,
                                         lambd, pub_pred, active_set)
        if any([loss > 0.1 for loss in losses]):
            print(losses)

        return ensemble_dist

    def reg_pred(self, output_dists):
        output = output_dists[0]
        for i in range(1, self.num_ensemble):
            output += output_dists[i] 
        return output / self.num_ensemble

    def print_priv_losses(self):
        print("-----------------------------")
        print("Per-instance privacy loss")
        for i, loss in enumerate(self.instance_priv_loss):
            print(f"Model {i}: {loss:.3f}")
        print("-----------------------------")
        print("Individual privacy loss")
        for i, loss in enumerate(self.personalized_priv_loss):
            print(f"Model {i}: {sum(loss):.3f}")
        print("-----------------------------")
        return
    
    def print_lambdas(self):
        for lambd in self.lambdas:
            print(lambd.item())

    def plot_individual_loss(self):
        x = [i for i in range(self.q_budget)]
        plt.plot(x, self.personalized_priv_loss[0])
        plt.savefig("priv_loss.png")