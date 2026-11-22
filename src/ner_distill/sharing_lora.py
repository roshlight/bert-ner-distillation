import copy
import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertConfig, BertForTokenClassification

from .sharing import apply_layer_sharing


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()

        self.base = base
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.base.requires_grad_(False)

    def forward(self, x):
        update = F.linear(F.linear(x, self.lora_a), self.lora_b)
        return self.base(x) + self.scale * update


def add_layer_specific_lora(model, rank=8, alpha=16):
    model.requires_grad_(False)
    new_layers = []
    for shared_layer in model.bert.encoder.layer:
        memo = {id(parameter): parameter for parameter in shared_layer.parameters()}
        memo.update({id(buffer): buffer for buffer in shared_layer.buffers()})
        layer = copy.deepcopy(shared_layer, memo)
        targets = [
            (name, module)
            for name, module in layer.named_modules()
            if isinstance(module, nn.Linear)
            and name.rsplit(".", 1)[-1] in {"query", "value"}
        ]

        for name, linear in targets:
            parent_name, child_name = name.rsplit(".", 1)
            parent = layer.get_submodule(parent_name)

            setattr(parent, child_name, LoRALinear(linear, rank=rank, alpha=alpha))

        new_layers.append(layer)

    model.bert.encoder.layer = nn.ModuleList(new_layers)
    model.config.layer_specific_lora = {"rank": rank, "alpha": alpha}

    return model


def load_lora_shared_student(path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    config_dict = dict(checkpoint["config"])
    lora_config = config_dict.pop("layer_specific_lora")

    config = BertConfig.from_dict(config_dict)
    model = BertForTokenClassification(config)

    apply_layer_sharing(model, layer_map=config.layer_sharing_map)
    add_layer_specific_lora(model, **lora_config)

    model.load_state_dict(checkpoint["state_dict"])
    return model
