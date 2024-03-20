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

class PMixED():
    def __init__(self, model_dirs, model_name, tokenizer,
                device="cpu", q_budget=1024, alpha=2, delta=1e-5,
                p=1.0, eps=0, beta=0., lambd=0, threshold=None, top_k=0,
                sigma=0.0
    ):
        self.device = device
        self.model_name = model_name
        self.q_budget = q_budget
        self.alpha = alpha 
        self.eps = eps
        self.priv_loss = 0
        self.p = p 
        self.delta = delta
        self.sigma = sigma
        print("Sigma value", self.sigma)
        self.model_dirs = model_dirs
        self.num_ensemble = len(self.model_dirs)
        self.lambd = lambd
        self.lambdas = np.array([0 for _ in range(self.num_ensemble)])
        self.lambda_history = []
        self.noisy_rd_history = []
        self.target = beta * self.alpha
        self.beta = beta
        self.threshold = threshold
        self.top_k = top_k
        self.num_noisy = 0 
        self.pub_model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                     pad_token_id=tokenizer.eos_token_id).to(
                                                     self.device)
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
    def subsample_eps(alpha, p, **kwargs):
        '''
        Make priv loss a parameter 
        '''
        return 1/(alpha-1) * np.log((1-p)**(alpha-1) * (1 + (alpha-1) * p) 
                                    + np.sum([comb(alpha, k) * (1-p)**(alpha-k) * p**k *
                                              np.exp((k-1) * PMixED.privacy_loss(alpha=alpha, **kwargs))
                                                for k in range(2, alpha+1)]
                                            )
                                        )
    @staticmethod
    def sample_privacy_loss(alpha, size, beta):
        if size == 1:
            return beta * alpha
        elif size == 0:
            return 0
        return (np.log((size - 1)/size + np.exp((alpha-1) * 4 * beta * alpha) / size ) 
                / (alpha-1))

    @staticmethod 
    def noisy_privacy_loss(alpha, size, lambd, sigma):
        return (lambd / size)**2 * alpha / sigma**2 

    @staticmethod
    def privacy_loss(alpha, size, beta=None, lambd=None, sigma=None):
        eps = 0 
        if beta is not None:
            eps += PMixED.sample_privacy_loss(alpha, size, beta)
        if lambd is not None:
            eps += PMixED.noisy_privacy_loss(alpha, size, lambd, sigma)
        return eps

    @staticmethod
    def renyiDiv(p, q, alpha=float('inf')):
        if alpha == float('inf'):
            RD = torch.log(torch.max(p/q))
        elif alpha == 1:
            RD = torch.sum(p*torch.log(p/q))
        else:
            inds = torch.nonzero(q**(alpha-1))
            RD = 1/(alpha-1)*torch.log(
                torch.sum((p[inds]**alpha)/(q[inds]**(alpha-1))))
        if torch.isnan(RD):
            RD = torch.log(torch.max(p/q))
        return RD 

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
            sub_eps = self.subsample_eps(self.alpha, self.p, size=size, beta=beta)
            return (sub_eps - self.eps/self.q_budget)
        return bisect(f, 0, 5, maxiter=20, disp=False)

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
            #self.lambda_history.append(0)
            self.priv_loss += self.subsample_eps(self.alpha,
                                            self.p,
                                            size=0,
                                            beta=self.beta
                                            )
            return output_dists[self.num_ensemble]

        if self.threshold is not None:
            '''
            Noisy Screening Mechanism:
            (1) Grab top-k entries V from public output distribution
            (2) Grab V entries from mixed distribution
            (3) Add noise to truncated mixed distribution
            (4) Renormalize probability distribution
            (5) Renyi Div between trunc mix dist and pub dist
            (6) if (4) > T then return pub dist else return mix dist
            (7) Record self.priv_loss 
            '''
            pub_logits = torch.log(output_dists[self.num_ensemble]).view(1, -1)
            pub_logits, idxs_remov = self.top_k_filtering(pub_logits, self.top_k)

            mix_dists = torch.stack([self.mix(output_dist, output_dists[-1], self.lambd)
                                  for output_dist in output_dists[:-1]]).mean(dim=0)
            priv_logits = torch.log(mix_dists).view(1, -1)
            priv_logits[idxs_remov] = -float("Inf")

            trunc_pub_output_dist = F.softmax(pub_logits[-1, :], dim=-1)
            trunc_priv_output_dist = F.softmax(priv_logits[-1, :], dim=-1)

            idx_noise = torch.nonzero(trunc_priv_output_dist, as_tuple=True)
            noise = torch.normal(0, self.sigma, size=(len(idx_noise), ))
            trunc_priv_output_dist[idx_noise] += noise

            min_output_dist = torch.abs(torch.min(trunc_priv_output_dist))
            norm_dp_output_dist = (trunc_priv_output_dist + min_output_dist) / (
                                    trunc_priv_output_dist + min_output_dist).sum()

            noisy_rd = self.rd_mollification(norm_dp_output_dist,
                                     trunc_pub_output_dist, self.alpha)
            self.noisy_rd_history.append(noisy_rd) 
            if noisy_rd > self.threshold:
                #print(f"Screening Mechanism invoked with {noisy_rd:.3f}")
                self.num_noisy += 1
                self.priv_loss += self.subsample_eps(self.alpha,
                                               self.p,
                                               size=len(sampled),
                                               lambd=self.lambd,
                                               sigma=self.sigma
                                               )
                return output_dists[self.num_ensemble] 
            self.target = self.beta * self.alpha
            self.priv_loss += self.subsample_eps(self.alpha,
                                                 self.p,
                                                 size=len(sampled),
                                                 beta=self.beta,
                                                 lambd=self.lambd,
                                                 sigma=self.sigma
                                                 )

        if self.threshold is None:
            self.target = self.beta_solver_bisection(len(sampled)) * self.alpha

        self.lambdas = np.array([self.lambda_solver_bisection(output_dists[i],
                                                output_dists[self.num_ensemble],
                                                )for i in sampled])
        self.lambda_history.append(np.mean([lambd for lambd in self.lambdas]))
        mixed_dists = [self.mix(output_dists[i], output_dists[self.num_ensemble], self.lambdas[lamb_i])
                       for lamb_i, i in enumerate(sampled)]
        output_dist = torch.stack(mixed_dists).mean(dim=0)
        return output_dist

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

    def mix(self, p, p_prime, lambd=0.5):
        return (lambd * p + (1-lambd) * p_prime)

    @staticmethod 
    def top_k_filtering(logits, top_k, filter_value=-float("Inf")):
        #  Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        return logits, indices_to_remove

    def print_priv_losses(self):
        print("-----------------------------")
        print(f"Total privacy loss {sum(self.priv_loss):.3f}")
        print("-----------------------------")
    
    def print_lambdas(self):
        print("-----------------------------")
        print(f"Average lambda value: {np.mean(self.lambda_history):.3f}")
        print("-----------------------------")
        print(f"Std lambda value: {np.std(self.lambda_history):.3f}")
        print("-----------------------------")
        print(f"Min lambda value: {np.min(self.lambda_history)}")
        print("-----------------------------")
        print(f"Max lambda value: {np.max(self.lambda_history):.3f}")
        print("-----------------------------")

    def print_noisy_rd(self):
        print("-----------------------------")
        print(f"Average noisy RD value: {np.mean(self.noisy_rd_history):.3f}")
        print("-----------------------------")
        print(f"Std noisy RD value: {np.std(self.noisy_rd_history):.3f}")
        print("-----------------------------")
        print(f"Min noisy RD value: {np.min(self.noisy_rd_history)}")
        print("-----------------------------")
        print(f"Max noisy RD value: {np.max(self.noisy_rd_history):.3f}")
        print("-----------------------------")

    def plot_lambdas(self):
        x = [i for i in range(len(self.lambda_history))]
        plt.figure().set_figwidth(15)
        plt.plot(x, self.lambda_history)
        plt.savefig(os.path.join("plts", "lambda_vals.png"))
        plt.clf()