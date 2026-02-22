#!/usr/bin/env python3
"""
GCP Service: Vertex AI Custom Job (TPU)
IAM Roles: roles/aiplatform.user
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm


@dataclass
class TrainBatch:
    inputs: torch.Tensor
    targets: torch.Tensor


def bce_tv_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    tv_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    tv = torch.mean(torch.abs(probs[..., 1:] - probs[..., :-1]))
    total = bce + tv_weight * tv
    return total, bce, tv


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: Iterable[TrainBatch],
    optimizer: torch.optim.Optimizer,
    tv_weight: float = 1.0,
) -> None:
    device = xm.xla_device()
    model.to(device)
    model.train()

    for batch in dataloader:
        inputs = batch.inputs.to(device)
        targets = batch.targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss, _, _ = bce_tv_loss(logits, targets, tv_weight=tv_weight)
        loss.backward()
        xm.optimizer_step(optimizer)


def run_training(
    model: torch.nn.Module,
    dataloader: Iterable[TrainBatch],
    optimizer: torch.optim.Optimizer,
    tv_weight: float = 1.0,
) -> None:
    """
    Adapter entrypoint to plug into SongFormer training.

    Replace the dataloader/model wiring with the SongFormer training pipeline.
    """

    train_one_epoch(model, dataloader, optimizer, tv_weight=tv_weight)
