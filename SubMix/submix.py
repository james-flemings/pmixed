# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import torch
import torch.nn as nn
import numpy as np
from transformers import GPT2Tokenizer
import scipy
from scipy.optimize import bisect
#from torch.distributions.normal import Normal
from itertools import chain, combinations

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

def logit2prob(logit):
    return torch.exp(logit)/(torch.sum(torch.exp(logit)))

class SubMix():
    def __init__(self, device, B, eps, public_model, ensemble,
        alpha=float('inf'), gamma=10, lambda_dec_factor=0.93, consumption_multiplier=1.0,
        lambda_solver='iteration', temp=1.0):
        self.alpha = alpha
        self.B = B
        self.queries_remaining = B
        self.eps_remaining = [eps]*len(ensemble)
        self.pairing = self._random_pairing(len(ensemble))
        self.STOP = False
        self.public_model = public_model
        self.ensemble = ensemble
        LMs = copy.copy(ensemble)
        LMs.insert(0,self.public_model)
        self.LMs = LMs
        self.temp = temp
        self.device = device
        self.queries_remaining = B
        self.tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        self.target_consumption = consumption_multiplier*eps/B
        assert lambda_dec_factor < 1.0, f'lambda_dec_factor should be less than 1'
        self.lambda_dec_factor = lambda_dec_factor
        self.gamma = gamma
        self.lambs = []
        self.epsilons = []
        self.lambda_solver = lambda_solver
    
    def compute_logits_at_context(self, context):
        if isinstance(context,str):
            context = torch.tensor(self.tokenizer.encode(x)).to(self.device)
        logit =  lambda model,x : model(x).logits.squeeze()
        L = [logit(lm,context) for lm in self.LMs]
        P = [nn.functional.softmax(logits, dim=1) for logits in L]
        return L, P

    def max_log_projection(self, P):
        #project all P into max-log ball around P0
        i = 1
        Q = [ P[0] ]
        for _ in range(len(P)-1):
            P_prime = torch.clone(P[i])
            max_log_dist = lambda P,Q : torch.max(torch.abs(
                torch.log(P) - torch.log(Q)))
            while max_log_dist(P_prime, P[0]) > self.gamma/2 + 1e-4:
                P_prime = torch.minimum(P_prime, np.exp(self.gamma/2)*P[0])
                P_prime = torch.maximum(P_prime, np.exp(-self.gamma/2)*P[0])
                P_prime = P_prime / torch.sum(P_prime)
            Q.append(P_prime)
            i += 1
        return Q 
        
    def mix(self, p, p_prime, lamb=0.5):
        mix = lamb*p + (1-lamb)*p_prime + 1e-20
        mix = mix/torch.sum(mix)
        assert (torch.sum(mix).item() -1.0)**2 <1e-10, 'this is not a pmf'
        return mix

    def renyi_dp(self, p ,q, alpha=float('inf')):
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
     
    def _lambda_solver_iteration(self, p, p_prime, p0):
        lamb = 1.0
        eps = float('inf')
        while eps > self.target_consumption:
            lamb = self.lambda_dec_factor*lamb
            mix_star = self.mix((p + p_prime)/2, p0, lamb=lamb)
            p_mix = self.mix(p, p0, lamb=lamb)
            p_prime_mix = self.mix(p_prime, p0, lamb=lamb)
            eps = max(self.renyi_dp(mix_star,p_mix, alpha=self.alpha),
                      self.renyi_dp(mix_star,p_prime_mix, alpha=self.alpha))
        return mix_star, lamb
    
    def _lambda_solver_bisection(self, p, p_prime, p0):
        def f(lamb):
            mix_star = self.mix((p + p_prime)/2, p0, lamb=lamb)
            p_mix = self.mix(p, p0, lamb=lamb)
            p_prime_mix = self.mix(p_prime, p0, lamb=lamb)
            eps = max(self.renyi_dp(mix_star,p_mix, alpha=self.alpha),
                      self.renyi_dp(mix_star,p_prime_mix, alpha=self.alpha))
            return eps - self.target_consumption
        lamb = 1.0 if f(1) <= 0.0 else bisect(f, 0, 1)
        return self.mix((p + p_prime)/2, p0, lamb=lamb), lamb
                
    def _get_lambda_from_dist_pair(self, p, p_prime, p0):
        if self.lambda_solver == 'iteration':
            rtn =  self._lambda_solver_iteration(p, p_prime, p0)
        elif self.lambda_solver == 'bisection':
            rtn = self._lambda_solver_bisection(p, p_prime, p0)
        return rtn
  
    def _mix (self, p, p_prime, p0, lamb):
        mix_star = self.mix((p + p_prime)/2, p0, lamb=lamb)
        p_mix = self.mix(p, p0, lamb=lamb)
        p_prime_mix = self.mix(p_prime, p0, lamb=lamb)
        return mix_star, max(self.renyi_dp(mix_star,p_mix, alpha=self.alpha),
                  self.renyi_dp(mix_star,p_prime_mix, alpha=self.alpha))

    def _get_eps(self, p, p_prime, p0, lamb, lamb_prime):
        mix_star = self.mix((p + p_prime)/2, p0, lamb=lamb)
        mix_L = self.mix(p, p0, lamb=lamb)
        mix_R = self.mix(p_prime, p0, lamb=(lamb_prime))
        return self.renyi_dp(mix_L, mix_R, alpha=self.alpha)
    
    
    def _random_pairing(self, k):
        randperm = torch.randperm(k) + 1 #add one to get (1,k) range
        pairing = []
        i = 0
        while i < len(randperm)-1:
            idx1 = randperm[i]
            idx2 = randperm[i+1]
            pairing.append( (idx1,idx2) )
            i += 2
        return pairing
        
    def _get_pairwise_lambdas(self, pairing, P):
        pairwise_lambdas = []
        mixes = []
        for i,j in pairing:
            #print(len(P))
            mix, lamb = self._get_lambda_from_dist_pair(P[i], P[j], P[0])
            pairwise_lambdas.append(lamb)
            mixes.append(mix)
        return pairwise_lambdas, mixes
    
    def _account_leakage(self, centroid, P, lamb, lambda_LOOs):
        eps_per_shard = []
        l = int(self.num_shards/2) #num logical shards
        for i in range(l):
            j,jj = self.pairing[i]
            P_prime = 0.5*(P[j] + P[jj])
            centroid_LOO = centroid*l/(l-1) - P_prime/(l-1)
            #print(len(lambda_LOOs))
            #print(i)
            lamb_prime = lambda_LOOs[i]
            eps = self._get_eps(centroid, centroid_LOO, P[0], lamb, lamb_prime)
            if self.symmetrize:
                _eps = self._get_eps(centroid_LOO, centroid, P[0], lamb_prime, lamb)
                eps = max(eps, _eps)
            eps_per_shard.append(eps)
            self.eps_remaining[i-1] -= eps
        return eps_per_shard
            
    def _compute_delta_lamb(self, lambdas):
        S = np.sum(lambdas)
        N = len(lambdas)
        S_mean = S/N
        S_LOOs = []
        for lamb in lambdas:
            S_LOOs.append( (S - lamb)/(N-1) )
        return S_LOOs
    
    def query(self, P):
        #L,P = self.compute_logits_at_context(context)
        if self.temp < 1.0:
            Z = lambda q : torch.sum(torch.exp(torch.log(q)/self.temp))
            P = [torch.exp(torch.log(p)/self.temp)/Z(p) for p in P]
        P = self.max_log_projection(P)
        if self.queries_remaining <= 0 or min(self.eps_remaining) <= 0:
            self.STOP = True
            P_out = P[0]
        else:                
            lambdas, mixes = self._get_pairwise_lambdas(self.pairing, P)
            P_out = sum(mixes)/len(mixes)
            epsilons = []
            l = len(P_out)
            for i,p in enumerate(mixes):
                p_prime = (P_out - p/l)*(l/(l-1))
                eps = self.renyi_dp(p_prime, P_out, alpha=self.alpha)
                eps = max(eps, self.renyi_dp(P_out, p, alpha=self.alpha))
                self.eps_remaining[i] -= eps
        self.queries_remaining -= 1
        return P_out
            
