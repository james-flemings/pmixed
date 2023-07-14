from transformers import AutoModelForCausalLM, AutoTokenizer
from ensemble import Ensemble
import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--num_ensemble", type=int, default=8)
parser.add_argument("--model_name", type=str, default="GPT2")
parser.add_argument("--data_subset", type=str, default="wikitext-103-raw-v1")
parser.add_argument("--device", type=str, default="cuda:7")
parser.add_argument("--seq_length", type=int, default=512)


def main():
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    pub_model = AutoModelForCausalLM.from_pretrained(args.model_name, pad_token_id=tokenizer.eos_token_id).to(args.device)

    model_paths = [os.path.join("models", f"lora-{args.model_name}-{i}-finetuned-{args.data_subset}") for i in range(args.num_ensemble)]
    priv_ensemble = Ensemble(model_paths, pub_model, args.device)
    
    seq_length = 526
    corpus = load_wikitext()
    val_loader = DataLoader(CorpusDataset(corpus['valid'], seq_length))
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(val_loader)):
            data = data.to(args.device)
            output_dists = priv_ensemble.pred_dist(data)
            for j in range(seq_length):
                priv_dist = priv_ensemble.priv_pred([output_dist[j]
                                                     for output_dist in output_dists])
                print(priv_dist)
                print(torch.sum(priv_dist))
                break
            break
        

def load_wikitext():
    corpus = dict()
    for dset in ["valid", "train", "test"]:
        corpus[dset] = torch.load(os.path.join("data", f"wikitext-103-{dset}-corpus.pt"))
    return corpus

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
