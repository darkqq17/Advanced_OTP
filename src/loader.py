import os
import requests
import torch
import torch.nn as nn
from typing import Union, List
from transformers.adapters import LoRAConfig

def freeze_parameters(model: nn.Module, trainable_name: Union[str, List[str]]):
    if isinstance(trainable_name, str):
        trainable_name = [trainable_name]
    for name, param in model.named_parameters():
        # train either vit adapter or language model
        if any(n in name for n in trainable_name):
            continue
            # print(name)
        else:
            param.requires_grad_(False)


def print_trainable_parameters(model: nn.Module, show_trained_param: bool = False):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if show_trained_param:
                print(name)
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )