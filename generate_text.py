import argparse
import logging
import torch
from torch.utils.data import DataLoader, Subset
import transformers
from tqdm import tqdm
import random
import numpy as np
import os
import csv
import collections
from pmixed import PMixED

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def calc_perplexity(encodings, cur_model):
    max_length = cur_model.config.n_positions
    stride = 512
    nlls_cur = []

    for i in range(0, encodings.size(0), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings[:, begin_loc:end_loc].to(args.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        target_ids[target_ids==cur_model.config.pad_token_id] = -100

        with torch.no_grad():
            outputs = cur_model(input_ids, labels=target_ids)
            nlls_cur.append(outputs[0] * trg_len)

    ppl_cur = torch.exp(torch.stack(nlls_cur).sum() / end_loc)

    return ppl_cur.item()


def main(args):
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id: 
        model = transformers.GPT2LMHeadModel.from_pretrained(args.model_name,
                                                            pad_token_id=tokenizer.pad_token_id
                                                            ).to(args.device)
    else:
        model = transformers.GPT2LMHeadModel.from_pretrained(args.model_name,
                                                             pad_token_id=tokenizer.eos_token_id
                                                            ).to(args.device)
    model.resize_token_embeddings(len(tokenizer))

    alpha = args.alpha
    epsilon = args.epsilon - np.log((alpha-1)/alpha) + (np.log(args.delta) + np.log(alpha))/(alpha-1)

    if args.subset == "None":
        args.subset = None

    model_dir = os.path.join("models", f"{args.num_ensemble}_ensemble")
    model_paths = None
    if args.subset == None:
        model_paths = [os.path.join(model_dir, f"lora-{args.model_name}-{i}-finetuned-{args.dataset}")
                    for i in range(args.num_ensemble)]
    else:
        model_paths = [os.path.join(model_dir, f"lora-{args.model_name}-{i}-finetuned-{args.subset}")
                    for i in range(args.num_ensemble)]
    model.eval()
    priv_ensemble = PMixED(model,
                           model_paths,
                           args.model_name,
                           tokenizer,
                           args.device,
                           q_budget=None,
                           alpha=alpha,
                           delta=args.delta,
                           p=args.p,
                           eps=epsilon,
                           beta=args.beta,
                           lambd=args.lambd,
                           threshold=args.threshold,
                           screen_top_k=args.screen_top_k,
                           sigma=args.sigma,
                           accounting_method=args.accounting_method
    )

    logger.info(args)
    def generate_text(prompt, seq_num, prompt_length):
        ppls_cur = []
        all_data = []
        input_ids = torch.tensor(prompt, device=args.device)#.view(1, -1)
        for _ in tqdm(range(seq_num), desc="Generating synthetic data"):
            output_sequence = priv_ensemble.priv_generate(
                input_ids=input_ids,
                max_length=args.seq_length,
                top_k=args.top_k,
            )
            '''
            priv_ensemble.lora_ensemble.set_adapter(f"lora-0")
            output_sequence = priv_ensemble.lora_ensemble.generate(
                input_ids=input_ids,
                max_length=args.seq_length,
                top_k=50,
                do_sample=True
            )
            ''' 
            generated_sequence = tokenizer.decode(output_sequence, skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=True)
            #ppl = calc_perplexity(output_sequence, model)
            #ppls_cur.append(ppl)
            labels, seq = generated_sequence[:prompt_length], generated_sequence[prompt_length:]
            seq = " ".join(seq.split())
            labels = labels.strip().split("\t")
            if seq:
                all_data.append([seq] + labels)
            print(f'Generated Text: "{seq}"')
            print("Privacy Loss:", PMixED.convert_to_aprox_dp(priv_ensemble.priv_loss, args.delta, args.alpha))
            print(f'Number of noisy screening used: {priv_ensemble.num_noisy}')
            priv_ensemble.num_noisy = 0
            priv_ensemble.print_noisy_rd()
            priv_ensemble.print_lambdas()
        return all_data, ppls_cur

    ppls_cur = []
    all_sequences = []
    title = 0
    with torch.no_grad():
        prompt_counter = collections.Counter()
        with open(args.input_training_file, encoding="utf-8") as rf:
            csv_reader = csv.reader(rf)
            title = next(csv_reader)
            label_column_index = [i for i, name in enumerate(title) if "label" in name]

            for line in csv_reader:
                prompt = "\t".join([line[idx] for idx in label_column_index]) + "\n\n"
                prompt_counter[prompt] += 1
        ratio_generation_training = args.total_sequences / sum(prompt_counter.values())

        for prompt_text in tqdm(prompt_counter, desc="Going through different types of prompts"):
            prompt = tokenizer(prompt_text)["input_ids"]
            num_seq_to_generate = round(prompt_counter[prompt_text] * ratio_generation_training)
            print(f'Prompt "{prompt_text}"')
            if num_seq_to_generate > 0:
                sequences, ppls = generate_text(prompt, num_seq_to_generate, len(prompt_text))
                all_sequences += sequences
                ppls_cur += ppls

    logger.info(f"Current PPL: %.2f±%.2f", np.mean(ppls_cur),np.std(ppls_cur))
    logger.info(f"Total generated sequences: %d", len(all_sequences))
    random.shuffle(all_sequences)

    output_name = f"{args.seq_len}_{args.dataset}_{args.epsilon}_dp_synthetic_data.csv" 
    output_path = os.path.join(args.output_dir, output_name)
    with open(output_path, 'w', encoding='utf-8') as wf:
        csv_writer = csv.writer(wf)
        csv_writer.writerow(title)
        for obj in all_sequences:
            if obj[0]:
                csv_writer.writerow(obj)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument(
        "--input_training_file",
        default=None,
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--seed", default=0)
    parser.add_argument("--total_sequences", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda:7")
    parser.add_argument("--use_cache", type=bool)

    parser.add_argument("--num_ensemble", type=int, default=8)
    parser.add_argument("--model_name", type=str, default="GPT2")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--accounting_method", type=str, default=None)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--alpha", type=int, default=3)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lambd", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.09)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--screen_top_k", type=int, default=100)

    args = parser.parse_args()
    main(args)