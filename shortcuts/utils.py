"""Utility functions for shortcuts."""

import json
import os
import random

import numpy as np
import torch
import torch.nn as nn


def init_weights(layer: nn.Module, pretraining: bool = False) -> None:
    """Create checkpoint with network(s) to be loaded in learning."""
    # initialize conv layers (unless pretraining is used)
    if not pretraining:
        if isinstance(layer, nn.Conv2d):
            nn.init.normal(layer.weight, std=0.1)
    # initialize linear and batchnorm layers
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(layer, nn.BatchNorm2d):
        nn.init.ones_(layer.weight)
    # initialize biases to zero
    if getattr(layer, 'bias', None) is not None:
        nn.init.zeros_(layer.bias)


def seed_everything(seed: int) -> None:
    """At least we tried."""
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_as_json(dictionary: dict, target: str) -> None:
    """Save a python object as JSON file."""
    with open(target, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=4)


def cast_tensor_float(x: torch.Tensor, digits: int = 4) -> str:
    """Cast tensor to float."""
    return format(float(x), f'.{digits}f')


def compute_entropy(predictions: torch.Tensor, eps: float = 0.0001) -> torch.Tensor:
    """Compute Shannon entropy with base-two log."""
    predictions_stabilized = predictions + eps
    return -(predictions * predictions_stabilized.log2()).sum(-1)
