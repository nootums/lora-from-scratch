import torch
import torch.nn as nn

class LoraModule(nn.Module):
    def __init__(self, orig_module:nn.modules.linear.Linear, r:int=8, alpha:int=1):
        assert type(orig_module)==nn.modules.linear.Linear, "Original module should be a Linear layer"
        assert type(r)==int, "r(rank) should be an integer"
        assert type(alpha)==int, "alpha should be an integer"
        super().__init__()
        self.alpha = alpha
        self.original_module = orig_module
        self.original_module.requires_grad = False
        orig_in_features = orig_module.in_features
        orig_out_features = orig_module.out_features
        self.lora_module = nn.Sequential(nn.Linear(orig_in_features, r, bias=False), nn.Linear(r, orig_out_features, bias=False))
        self.lora_module[0].weight = nn.Parameter(self.lora_module[0].weight)
        self.lora_module[1].weight = nn.Parameter(self.lora_module[1].weight)
        return
    
    def forward(self, x, *args, **kwargs):
        outs = self.original_module(x) + self.alpha*self.lora_module(x)
        return outs
    
    def set_alpha(self, new_alpha:int):
        assert type(new_alpha)==int, "New alpha value must be int"
        self.alpha = new_alpha
        return
    
