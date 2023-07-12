import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

class Ensemble():
    def __init__(self, device, q_budget, alpha, eps, pub_model, model_dirs, device):
        self.device = device
        self.q_budget = q_budget
        self.alpha = alpha
        self.eps = eps
        self.pub_model = pub_model
        self.model_dirs = model_dirs
        self.device = device
        self.models = []
        for dir in self.model_dirs:
            config = PeftConfig.from_pretrained(dir)
            model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
            self.models += PeftModel.from_pretrained(model, dir).to(self.device)


    @staticmethod
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

