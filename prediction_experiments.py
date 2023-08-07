import torch
import torch.nn as nn
from torcheval.metrics.text import Perplexity
from transformers import AutoModelForCausalLM, AutoTokenizer
import os 
import argparse
from ensemble import Ensemble
from datasets import load_dataset
from training_ensemble import group_texts, wiki_tokenize_function
from peft import PeftModel
import copy
import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--num_ensemble", type=int, default=8)
parser.add_argument("--model_name", type=str, default="GPT2")
parser.add_argument("--dataset", type=str, default="wikitext")
parser.add_argument("--data_subset", type=str, default="wikitext-103-raw-v1")
parser.add_argument("--device", type=str, default="cuda:6")
parser.add_argument("--seq_length", type=int, default=512)
parser.add_argument("--query_budget", type=int, default=1024)

def main():
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model_paths = [os.path.join("models", f"lora-{args.model_name}-{i}-finetuned-{args.data_subset}")
                    for i in range(args.num_ensemble)]
    priv_ensemble = Ensemble(model_paths,
                             args.model_name,
                             tokenizer,
                             args.device,
                             q_budget=args.query_budget)


    fine_tuned_model_dir = os.path.join("models", f"lora-{args.model_name}-finetuned-{args.data_subset}")
    fine_tuned_model = PeftModel.from_pretrained(copy.deepcopy(priv_ensemble.pub_model),
                                                 fine_tuned_model_dir,
                                                 pad_token_id=tokenizer.eos_token_id).to(
                                                 args.device)

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
    ensemble_neg_log_likelihood= []

    for i in tqdm.tqdm(range(len(test_data))):
        data = test_data[i]['input_ids'].to(args.device)
        labels = test_data[i]['labels'].to(args.device)
        with torch.no_grad():
            pub_output_logits = priv_ensemble.pub_model(data, labels=labels).logits
            fine_tuned_output_logits = fine_tuned_model(data, labels=labels).logits
            
            output_dists = priv_ensemble.pred_dist(data)
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

    pre_trained_ppl = torch.exp(torch.stack(pub_neg_log_likelihood)).mean()
    fine_tuned_ppl = torch.exp(torch.stack(fine_tuned_neg_log_likelihood)).mean()
    ensemble_ppl = torch.exp(torch.stack(ensemble_neg_log_likelihood)).mean()

    print(f"Perplexity score for Pre-Trained Model: {pre_trained_ppl:.2f}")
    print(f"Perplexity score for Fine-Tuned Model: {fine_tuned_ppl:.2f}")
    print(f"Perplexity score for Ensemble Model: {ensemble_ppl:.2f}")

    priv_ensemble.print_priv_losses()
    priv_ensemble.plot_individual_loss()

def calc_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = nn.CrossEntropyLoss()
    return loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

if __name__ == "__main__":
    main()