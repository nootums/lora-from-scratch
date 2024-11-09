import torch
import torch.nn as nn
import transformers

class LoraModule(nn.Module):
    def __init__(self, orig_module:nn.modules.linear.Linear, r:int=8, alpha:float=1.0):
        assert type(orig_module)==nn.modules.linear.Linear, "Original module should be a Linear layer"
        assert type(r)==int, "r(rank) should be an integer"
        assert type(alpha)==float, "alpha should be a float"
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
    
    def set_alpha(self, new_alpha:float):
        assert type(new_alpha)==float, "New alpha value must be a float"
        self.alpha = new_alpha
        return


def convert_model_to_lora_model(model:transformers.modeling_utils.PreTrainedModel):
    for name, module in model.named_parameters():
        module.requires_grad = False
        if module.ndim==1:
            module.data = module.data.to(torch.float32)
            continue
        names = name.split('.')[:-1]
        module_pointer = model
        module_pointer_parent = None
        for layer in names:
            module_pointer_parent = module_pointer
            module_pointer = getattr(module_pointer, layer)
            if type(module_pointer)==LoraModule:
                break
        if type(module_pointer)==LoraModule:
            continue
        if type(module_pointer)==nn.modules.linear.Linear:
            lora_module = LoraModule(module_pointer)
            setattr(module_pointer_parent, names[-1], lora_module)
    return

def change_lora_alpha(model:transformers.modeling_utils.PreTrainedModel, new_alpha):
    for name, module in model.named_parameters():
        names = name.split('.')[:-1]
        module_pointer = model
        for layer in names:
            module_pointer = getattr(module_pointer, layer)
            if type(module_pointer)==LoraModule:
                break
        if type(module_pointer)==LoraModule:
            module_pointer.set_alpha(new_alpha)