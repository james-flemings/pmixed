# priv_pred_ensemble

An algorithm involving Low-Rank Adaptation (LoRA), subsample-and-aggregate, and RDP filters to achieve RDP prediction.

To run the code, use the following command: 

```bash
torchrun --nproc_per_node=num_gpus training_ensemble.py
```

where num_gpus is the total number of gpus you currently have.

```bash
torchrun --nproc_per_node=num_gpus training_ensemble.py --training_type="dpsgd" 
```
