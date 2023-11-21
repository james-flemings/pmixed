# PMixED: Private Mixing of Ensemble Distributions 

PMixED is a differentially private next token prediction protocol that mixes the output distribution of privacy-sensitive fine-tuned models and of a public model.  

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
Fine-tuned models used in the paper can be accessed [here](https://drive.google.com/file/d/1QW70rCqhSPd7KyzrGZk8noK6v5SRArlr/view?usp=sharing). Be sure to untar the file in the root directory of this directory. However, if you decide to train the ensemble from scratch, then use the following command to reproduce our models:

```bash
python -m torch.distributed.run --nproc_per_node=num_gpus fine_tune_ensemble.py \
    --model_name=GPT2 \
    --dataset=wikitext \
    --subset=wikitext-103-raw-v1 \
    --num_ensemble=100 \
    --epochs=15 \
    --lora_r=4 \
    --lora_alpha=32 \
    --lora_dropout=0.1 \
    --block_size=512 \
    --learning_rate=2e-4 \
    --weight_decay=0.01 \
    --batch_size=8 \
    --dp_batch_size=256 \
    --training_type=samp-agg \
    --max_grad_norm=1. \
    --epsilon=8. \
    --delta=1e-5 \
    --num_gpus=num_gpus \
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