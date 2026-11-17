"""Model definitions, dataset loading, and parameter/MAC helpers for Lab 7."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from thop import profile
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for deterministic runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class OversizedCNN(nn.Module):
    """Oversized 3-block CNN baseline/teacher (~1.01M parameters) for Fashion-MNIST."""

    def __init__(
        self,
        channels: tuple[int, int, int] = (24, 48, 160),
        fc_hidden: int = 640,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.channels = (c1, c2, c3)
        self.fc_hidden = fc_hidden
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7 -> 3x3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 3 * 3, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class StudentCNN(nn.Module):
    """Compact 2-block CNN student (~99.6k parameters) for Fashion-MNIST."""

    def __init__(
        self,
        conv_channels: tuple[int, int] = (16, 32),
        fc_hidden: int = 60,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        c1, c2 = conv_channels
        self.conv_channels = (c1, c2)
        self.fc_hidden = fc_hidden
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * 7 * 7, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_prunable_weights(model: nn.Module) -> tuple[int, int]:
    """Return (nonzero_weights, total_weights) across Conv2d and Linear weight matrices."""
    nonzero = 0
    total = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.detach()
            nonzero += int(torch.count_nonzero(w).item())
            total += int(w.numel())
    return nonzero, total


def count_all_nonzero_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (nonzero_params, total_params) across all parameters (weights + biases)."""
    nonzero = 0
    total = 0
    for p in model.parameters():
        nonzero += int(torch.count_nonzero(p.detach()).item())
        total += int(p.numel())
    return nonzero, total


def compute_macs_and_params(
    model: nn.Module, input_shape: tuple[int, int, int, int] = (1, 1, 28, 28)
) -> tuple[float, int]:
    """Compute dense MACs via thop on a clean deepcopy (avoids mutating live model buffers)."""
    model_copy = copy.deepcopy(model).eval()
    dummy = torch.randn(*input_shape)
    macs, params = profile(model_copy, inputs=(dummy,), verbose=False)
    return float(macs), int(params)


def load_fashion_mnist(
    root: str | Path = "lab03/src/data",
    train_size: int = 55000,
    val_size: int = 5000,
    batch_size: int = 256,
    eval_batch_size: int = 256,
    seed: int = 42,
    download: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load Fashion-MNIST into normalized TensorDatasets with deterministic 55k/5k/10k splits."""
    mean, std = 0.2860, 0.3530

    raw_train = datasets.FashionMNIST(root=str(root), train=True, download=download)
    raw_test = datasets.FashionMNIST(root=str(root), train=False, download=download)

    x_train_full = (raw_train.data.float().unsqueeze(1) / 255.0 - mean) / std
    y_train_full = raw_train.targets.long()
    full_ds = TensorDataset(x_train_full, y_train_full)

    x_test = (raw_test.data.float().unsqueeze(1) / 255.0 - mean) / std
    y_test = raw_test.targets.long()
    test_ds = TensorDataset(x_test, y_test)

    split_gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=split_gen)

    loader_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=loader_gen
    )
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate_model(
    model: nn.Module, loader: DataLoader, criterion: nn.Module | None = None
) -> dict[str, float]:
    """Evaluate average loss and top-1 accuracy (%) on a DataLoader."""
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        logits = model(x)
        loss = criterion(logits, y)
        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        correct += int((logits.argmax(dim=1) == y).sum().item())
        total += bs

    return {
        "loss": total_loss / total,
        "acc": 100.0 * correct / total,
    }


def train_epoch_standard(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> dict[str, float]:
    """Train one standard supervised epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        correct += int((logits.argmax(dim=1) == y).sum().item())
        total += bs

    return {
        "loss": total_loss / total,
        "acc": 100.0 * correct / total,
    }
