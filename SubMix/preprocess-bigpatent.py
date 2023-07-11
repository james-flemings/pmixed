#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import torch
import tqdm
from transformers import GPT2Tokenizer
from datasets import load_dataset
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
dataset = load_dataset("big_patent","g")

from joblib import Parallel, delayed
def big_patent_preprocess(n_jobs, batchsize=100):
    from datasets import load_dataset
    from gpt2.data.dataset import CorpusDataset, UserLvlDataset
    
    def text2tensor(dset, i):
        for j in range(i,batchsize+i):
            if j >= len(dataset[dset]):
                break
            doc = dataset[dset][j]
            txt = 'abstract : ' + doc['abstract']
            txt += ' description : ' + doc['description']
            txt = torch.tensor(tokenizer.encode(txt))
            torch.save(txt, f'bigpatent-{dset}-{j}.pt')

    for dset in ['train', 'test', 'validation']:
        method = lambda i : text2tensor(dset,i)
        Parallel(n_jobs=n_jobs)(delayed(method)(i) for i in range(0,len(dataset[dset]),batchsize))
        
n_cores = 32
big_patent_preprocess(n_cores) #replace with whatever # of cores your machine has

def _tensorloading(dset):
    tensors = []
    for i,t_path in enumerate(glob.glob(f'bigpatent-{dset}-*.pt')):
        tensors.append(torch.load(t_path))
        if (i+1) % 1000 == 0:
            print(f'Procssed {dset} tensor file # {i+1}...')
    return torch.cat(tensors)

for dset in ['train', 'test', 'validation']:
    corpus = _tensorloading(dset)
    torch.save(corpus,f'bigpatent-{dset}_full.pt')

       
