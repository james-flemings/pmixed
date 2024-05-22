# PMixED: Private Mixing of Ensemble Distributions 

PMixED is a differentially private next token prediction protocol that mixes the output distribution of privacy-sensitive fine-tuned models and the output distribution of a public model.  

## Evironment Setup
We used Python3.10 in our implementation. Run the following lines to set up the evironment: 

```bash
sudo apt install python3.10
sudo apt install python3.10-venv
python3.10 -m ensurepip --upgrade
python3.10 -m venv venv
source venv/bin/activate
python3.10 -m pip install -r requirements.txt
```

## Ensemble/DP model Setup
Fine-tuned models used in the paper can be accessed [here](https://drive.google.com/file/d/1v4Yp1AdofrXLqmb-ip4iXcYFHk9x_yt6/view?usp=drive_link). Be sure to untar the file in the root directory of this directory. However, if you decide to train the ensemble from scratch, then use the following command to reproduce our models:

```bash
python -m torch.distributed.run --nproc_per_node=8 fine_tune_ensemble.py \
    --model_name=GPT2 \
    --dataset=yelp \
    --data_path=data/yelp_data \
    --num_ensemble=25 \
    --epochs=6 \
    --lora_r=4 \
    --lora_alpha=32 \
    --lora_dropout=0. \
    --block_size=128 \
    --learning_rate=8e-5 \
    --weight_decay=0.01 \
    --batch_size=32 \
    --dp_batch_size=256 \
    --training_type=samp-agg \
    --max_grad_norm=1. \
    --epsilon=8. \
    --delta=1e-5 \
    --num_gpus=8 \
    --num_proc=64
```

To reproduce the DP-SGD model, set ```--training_type=dpsgd```. To train on the One Billion Word dataset, set ```--dataset=lm1b --subset=None```.

## Private Prediction Experiments
To reproduce our comparison result, run the following command
```bash
python prediction_experiments.py \
    --num_ensemble=80 \
    --model_name=GPT2 \
    --dataset=wikitext \
    --subset=wikitext-103-raw-v1 \
    --device=cpu \
    --seq_length=512 \
    --epsilon=8.0 \
    --alpha=3 \
    --delta=1e-5 \
    --p=0.03 \
    --iters=32
```

And for the Ablation Study on the hyperparameters: ```python hyperparameter_experiments.py```.


## Noisy Screening
To reproduce our comparison result, run the following command
```bash
python prediction_experiments.py \
    --num_ensemble=100 \
    --model_name=GPT2 \
    --dataset=wikitext \
    --subset=wikitext-103-raw-v1 \
    --device=cuda:7 \
    --accounting_method=Dependent \
    --seq_length=512 \
    --epsilon=8.0 \
    --query_budget=1024 \
    --alpha=18 \
    --delta=1e-5 \
    --p=0.03 \
    --threshold=4.5 \
    --sigma=1e-2 \
    --lambd=1e-4 \
    --beta=0.2 \
    --screen_top_k=60 \
    --iters=1
```