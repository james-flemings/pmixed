import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import os 
import argparse
from ensemble import Ensemble
from datasets import load_dataset
from training_ensemble import group_texts, wiki_tokenize_function
from peft import PeftModel
import copy
import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--num_ensemble", type=int, default=8)
parser.add_argument("--model_name", type=str, default="GPT2")
parser.add_argument("--dataset", type=str, default="wikitext")
parser.add_argument("--data_subset", type=str, default="wikitext-103-raw-v1")
parser.add_argument("--device", type=str, default="cuda:6")
parser.add_argument("--seq_length", type=int, default=512)
parser.add_argument("--query_budget", type=int, default=1024)
parser.add_argument("--target_multiplier", type=float, default=1.0)
parser.add_argument("--epsilon", type=float, default=2.0)

def main():
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_dir = os.path.join("models", f"{args.num_ensemble}_ensemble")

    model_paths = [os.path.join(model_dir, f"lora-{args.model_name}-{i}-finetuned-{args.data_subset}")
                    for i in range(args.num_ensemble)]
    priv_ensemble = Ensemble(model_paths,
                             args.model_name,
                             tokenizer,
                             args.device,
                             q_budget=args.query_budget,
                             eps=args.epsilon,
                             target_mult=args.target_multiplier)


    fine_tuned_model_dir = os.path.join("models", f"lora-{args.model_name}-finetuned-{args.data_subset}")
    fine_tuned_model = PeftModel.from_pretrained(copy.deepcopy(priv_ensemble.pub_model),
                                                 fine_tuned_model_dir,
                                                 pad_token_id=tokenizer.eos_token_id).to(
                                                 args.device)
    dp_fine_tuned_model = torch.load(os.path.join("models", f"lora-{args.model_name}-{args.epsilon}-dp-finetuned-{args.data_subset}.pt")).to(args.device)

    seq_length = 512
    dataset = load_dataset(args.dataset, args.data_subset)

    tokenize_function = wiki_tokenize_function if args.dataset == "wikitext" else None
    remove_columns = ["text"] if args.dataset == "wikitext" else None
    tokenized_dataset = dataset.map(tokenize_function,
                                    fn_kwargs={"tokenizer": tokenizer},
                                    batched=True,
                                    num_proc=4,
                                    remove_columns=remove_columns
                                    )
    lm_dataset = tokenized_dataset.map(
        group_texts,
        fn_kwargs={"block_size": seq_length},
        batched=True,
        num_proc=4
    ) 

    test_data = lm_dataset['test']
    test_data.set_format(type="torch")

    pub_neg_log_likelihood = []
    fine_tuned_neg_log_likelihood = []
    dp_fine_tuned_neg_log_likelihood = []
    ensemble_neg_log_likelihood= []
    test_loader = DataLoader(test_data)

    for i, data in enumerate(tqdm.tqdm(test_loader)):
        labels = data['labels'].to(args.device)
        input_ids = data['input_ids'].to(args.device)
        with torch.no_grad():
            pub_output_logits = priv_ensemble.pub_model(input_ids).logits
            fine_tuned_output_logits = fine_tuned_model(input_ids).logits
            dp_fine_tuned_output_logits = dp_fine_tuned_model(input_ids).logits
            
            output_dists = priv_ensemble.pred_dist(input_ids)
            ensemble_logits = []

            if i < args.query_budget // seq_length:
                for j in range(seq_length):
                    token_softmax = [output_dist[j] for output_dist in output_dists]
                    ensemble_output_dist = priv_ensemble.priv_pred(token_softmax)
                    ensemble_logits.append(torch.log(ensemble_output_dist))

                ensemble_logits = torch.stack(ensemble_logits)
                ensemble_neg_log_likelihood.append(calc_loss(ensemble_logits, labels))
            else:
                break

        pub_neg_log_likelihood.append(calc_loss(pub_output_logits, labels))
        fine_tuned_neg_log_likelihood.append(calc_loss(fine_tuned_output_logits, labels))
        dp_fine_tuned_neg_log_likelihood.append(calc_loss(dp_fine_tuned_output_logits, labels))

    pre_trained_ppl = torch.exp(torch.stack(pub_neg_log_likelihood))
    fine_tuned_ppl = torch.exp(torch.stack(fine_tuned_neg_log_likelihood))
    dp_fine_tuned_ppl = torch.exp(torch.stack(dp_fine_tuned_neg_log_likelihood))
    ensemble_ppl = torch.exp(torch.stack(ensemble_neg_log_likelihood))

    print(f"Perplexity score for Pre-Trained Model: {pre_trained_ppl.mean():.2f}")
    print(f"Perplexity score for Fine-Tuned Model: {fine_tuned_ppl.mean():.2f}")
    print(f"Perplexity score for DP-Fine-Tuned Model: {dp_fine_tuned_ppl.mean():.2f}")
    print(f"Perplexity score for Ensemble Model: {ensemble_ppl.mean():.2f}")

    priv_ensemble.print_priv_losses()
    priv_ensemble.print_lambdas()
    priv_ensemble.plot_individual_loss()
    priv_ensemble.plot_lambdas()
    #plot_ppl(ensemble_ppl)

def calc_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = nn.CrossEntropyLoss()
    return loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

def plot_ppl(ppl_scores):
    x = [i for i in range(len(ppl_scores))]
    plt.plot(x, ppl_scores.cpu())
    plt.savefig(os.path.join("plt", "ensemble_perplexity_scores.png"))
    plt.clf()


if __name__ == "__main__":
    main()