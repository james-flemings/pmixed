from transformers import AutoModelForCausalLM, AutoTokenizer
from ensemble import Ensemble
import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torcheval.metrics.text import Perplexity
import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--num_ensemble", type=int, default=8)
parser.add_argument("--model_name", type=str, default="GPT2")
parser.add_argument("--data_subset", type=str, default="wikitext-103-raw-v1")
parser.add_argument("--device", type=str, default="cuda:7")
parser.add_argument("--seq_length", type=int, default=512)
parser.add_argument("--query_budget", type=int, default=1048)


def main():
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    pub_model = AutoModelForCausalLM.from_pretrained(args.model_name, pad_token_id=tokenizer.eos_token_id).to(args.device)

    model_paths = [os.path.join("models", f"lora-{args.model_name}-{i}-finetuned-{args.data_subset}") for i in range(args.num_ensemble)]
    priv_ensemble = Ensemble(model_paths,
                             pub_model,
                             args.device,
                             q_budget=args.query_budget)
    
    seq_length = 526
    corpus = load_wikitext()
    val_loader = DataLoader(CorpusDataset(corpus['valid'], seq_length))
    priv_losses = []
    pub_losses = []
    fine_tuned_losses = []
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(val_loader)):
            metric = Perplexity()
            data = data.to(args.device)
            output_dists = priv_ensemble.pred_dist(data)
            pub_dists = pub_model(data).logits.squeeze().to('cpu')

            priv_dists = []
            fine_tuned_dists = []
            for j in range(seq_length):
                inp = [output_dist[j] for output_dist in output_dists]
                fine_tuned_dist = priv_ensemble.reg_pred(inp)
                priv_dist = priv_ensemble.priv_pred(inp)
                
                priv_dists.append(torch.log(priv_dist))
                fine_tuned_dists.append(torch.log(fine_tuned_dist))

            priv_dists = torch.stack(priv_dists).unsqueeze(0)
            fine_tuned_dists = torch.stack(fine_tuned_dists).unsqueeze(0)
            pub_dists = pub_dists.unsqueeze(0)

            labels = data.squeeze(0).unsqueeze(-1)
            metric.update(priv_dists, data.to('cpu'))
            priv_losses.append(metric.compute().item())
            metric.update(pub_dists, data.to('cpu'))
            pub_losses.append(metric.compute().item())
            metric.update(fine_tuned_dists, data.to('cpu'))
            fine_tuned_losses.append(metric.compute().item())

            #priv_dists = torch.stack(priv_dists)
            #fine_tuned_dists = torch.stack(fine_tuned_dists)

            #priv_losses.append(XHeval(priv_dists, data.to('cpu')))
            #fine_tuned_losses.append(XHeval(fine_tuned_dists, data.to('cpu')))
            #pub_losses.append(XHeval(pub_dists, data.to('cpu')))
            #priv_ensemble.print_priv_budgets()

            if i >= int(args.query_budget / seq_length):
                break

        print(f"Ensemble Val Loss: {np.mean(priv_losses):.4f} ")
        print(f"Fine-Tuned Val Loss: {np.mean(fine_tuned_losses):.4f} ")
        print(f"Pre-Trained Val Loss: {np.mean(pub_losses):.4f} ")

def load_wikitext():
    corpus = dict()
    for dset in ["valid", "train", "test"]:
        corpus[dset] = torch.load(os.path.join("data", f"wikitext-103-{dset}-corpus.pt"))
    return corpus

def XHeval(lm_logits, labels):
    #computes cross entropy
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    # Flatten the tokens
    loss = nn.CrossEntropyLoss()
    output = loss(shift_logits, shift_labels).item()
    return output

class CorpusDataset(Dataset):
    def __init__(self, corpus, seqlen):
        super().__init__()
        self.corpus = corpus
        self.seqlen = seqlen

    def __len__(self):
        return int(len(self.corpus) / self.seqlen)

    def __getitem__(self, item):
        idx = item * self.seqlen
        return self.corpus[idx : idx + self.seqlen]

if __name__ == "__main__":
    main()
