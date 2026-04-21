from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from anti_spoofing.registries import LOSS_REGISTRY


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


@LOSS_REGISTRY.register("weighted_cce")
def build_weighted_cce(device, class_weight=None, focal_gamma: float = 2.0):
    weight = class_weight
    if weight is None:
        weight = torch.FloatTensor([0.1, 0.9]).to(device)
    return nn.CrossEntropyLoss(weight=weight)


@LOSS_REGISTRY.register("cce")
def build_cce(device, class_weight=None, focal_gamma: float = 2.0):
    return nn.CrossEntropyLoss()


@LOSS_REGISTRY.register("focal")
def build_focal(device, class_weight=None, focal_gamma: float = 2.0):
    weight = class_weight
    if weight is None:
        weight = torch.FloatTensor([0.1, 0.9]).to(device)
    return FocalLoss(weight=weight, gamma=focal_gamma)


def build_loss(loss_name: str, device, class_weight=None, focal_gamma: float = 2.0):
    return LOSS_REGISTRY.get(loss_name.lower())(
        device=device, class_weight=class_weight, focal_gamma=focal_gamma
    )

