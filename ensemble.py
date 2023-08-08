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
        self.lambdas = np.array([1/4 for _ in range(self.num_ensemble // 2)])
        self.instance_priv_loss = [0 for _ in range(self.num_ensemble // 2)]
        self.personalized_priv_loss = [[] for _ in range(self.num_ensemble // 2)]
        self.pairings = [(i, i+1) for i in range(0, self.num_ensemble, 2)]
        self.target = target_mult * self.eps 
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
            #output_dists.append(logits)
        logits = self.pub_model(context).logits.squeeze()
        output_dists.append(nn.functional.softmax(logits, dim=1))
        #output_dists.append(logits)
        return output_dists 

    def calc_per_inst_priv_loss(self, p, mixed_dist, lambd, pub_pred):
        N = len(mixed_dist) 
        for i, mix_dist in enumerate(mixed_dist):
            lambd_i = np.mean(self.lambdas[:i] + self.lambdas[i+1:]).item()
            q = (p - mix_dist) * N / (N - 1)
            r = p + (1 - lambd) * pub_pred
            q = q / lambd 
            q += pub_pred
            self.instance_priv_loss[i] += max(Ensemble.renyiDiv(r, q, self.alpha),
                                              Ensemble.renyiDiv(q, r, self.alpha)
                                              )

    def calc_indiv_priv_loss(self, p, mixed_dist, lambd, pub_pred, active_set):
        losses = []
        N = len(mixed_dist)
        for i, mix_dist in enumerate(mixed_dist):
            lambd_i = np.mean(self.lambdas[:i] + self.lambdas[i+1:]).item()
            val, ind = torch.max(mix_dist, 0)
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
            r = mix_dist + (1 - lambd) * pub_pred
            q = copy.deepcopy((1 - lambd_i) * pub_pred)
            r[ind] += (N-1) / N * lambd 
            q[ind] += lambd_i
            loss = max(Ensemble.renyiDiv(r, q, self.alpha).cpu(),
                       Ensemble.renyiDiv(q, r, self.alpha).cpu()
            )
            self.personalized_priv_loss[i].append(loss)
            losses.append(loss)
        return losses


    def priv_pred(self, output_dists):
        #self.lambdas = [self.lambda_solver_bisection(output_dists[i].cpu(),
        #                                             output_dists[j].cpu(),
        #                                             output_dists[self.num_ensemble].cpu()
        #                                             ) for i, j in self.pairings]
        self.lambdas = [0.67 for _ in range(self.num_ensemble // 2)]
        lambd = np.mean(np.array(self.lambdas)).item()
        self.lambda_history.append(lambd)

        mixed_dists = [lambd * (output_dists[i] + output_dists[j]) / (self.num_ensemble) 
                        for i, j in self.pairings]
        ensemble_dist = sum(mixed_dists) 
        pub_pred = output_dists[self.num_ensemble]
        ensemble_dist += pub_pred 
        active_set = [i for i in range(self.num_ensemble // 2)]
        #self.calc_per_inst_priv_loss(ensemble_dist-pub_pred, mixed_dists, lambd, pub_pred)
        losses = self.calc_indiv_priv_loss(ensemble_dist-pub_pred, mixed_dists,
                                         lambd, pub_pred, active_set)

        return ensemble_dist
        #active_set = [i for i, loss in enumerate(losses) if loss < self.eps / (self.q_budget)]

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
        #if any([loss > 0.1 for loss in losses]):
        #    print(losses)
        return ensemble_dist
    
    def lambda_solver_bisection(self, p, p_prime, pub_model):
        def f(lambd):
            mix_star = self.mix((p + p_prime)/2, pub_model, lambd=lambd)
            p_mix = self.mix(p, pub_model, lambd)
            p_prime_mix = self.mix(p_prime, pub_model, lambd)

            q = self.mix(p/2, pub_model, lambd)
            r = self.mix(p_prime/2, pub_model, lambd)

            q_pub = copy.deepcopy((1 - lambd) * pub_model)
            r_pub = copy.deepcopy((1 - lambd) * pub_model)

            _, q_ind = torch.max(q, 0)
            _, r_ind = torch.max(r, 0)

            q[q_ind] += lambd/2
            q_pub[q_ind] += lambd

            r[r_ind] += lambd/2
            r_pub[r_ind] += lambd

            eps = max(self.renyiDiv(q, q_pub, alpha=self.alpha),
                      self.renyiDiv(r, r_pub, alpha=self.alpha)
                      )

            #eps = max(self.renyiDiv(mix_star, p_mix, alpha=self.alpha),
            #          self.renyiDiv(mix_star, p_prime_mix, alpha=self.alpha)
            #          )
            return eps - self.target 
        lambd = 0.99 if f(0.99) <= 0.0 else bisect(f, 0, 1, maxiter=5, disp=False)
        return lambd

    def mix(self, p, p_prime, lambd=0.5):
        mix_out = lambd * p + (1-lambd) * p_prime + 1e-20
        mix_out = mix_out/torch.sum(mix_out)
        assert (torch.sum(mix_out).item() - 1.0)**2 < 1e-10, "This is not a pmf"
        return mix_out


    def reg_pred(self, output_dists):
        output = output_dists[0]
        for i in range(1, self.num_ensemble):
            output += output_dists[i] 
        return output / self.num_ensemble

    def print_priv_losses(self):
        #print("-----------------------------")
        #print("Per-instance privacy loss")
        #for i, loss in enumerate(self.instance_priv_loss):
        #    print(f"Model {i}: {loss:.3f}")
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