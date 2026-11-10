"""Best model architecture selected from Lab 3 (CNN on FashionMNIST)."""

import torch.nn as nn


class CNN(nn.Module):
    """Compact 2-block convolutional classifier from Lab 3 (~20.5k params)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    """Factory function to construct model from config name."""
    if name.upper() == "CNN":
        return CNN(num_classes=num_classes)
    raise ValueError(f"Unsupported model name: {name!r}")
