#!/usr/bin/env python3
"""
Train a small CNN on CIFAR-10 and save a checkpoint for ONNX export.

Example:
  python train_model.py --epochs 5 --out artifacts/last.pt
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


@dataclass
class TrainMeta:
    dataset: str = "CIFAR10"
    num_classes: int = 10
    image_c: int = 3
    image_h: int = 32
    image_w: int = 32
    cifar_mean: Tuple[float, float, float] = CIFAR_MEAN
    cifar_std: Tuple[float, float, float] = CIFAR_STD


class SmallCnn(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == targets).float().mean().item())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", type=str, default="artifacts/last.pt")
    args = p.parse_args()

    torch.manual_seed(0)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
    test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCnn().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * x.size(0)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += int((pred == y).sum().item())
                total += int(y.numel())

        print(
            f"epoch {epoch:02d}  train_loss={total_loss/len(train_ds):.4f}  test_acc={correct/total:.4f}"
        )

    meta = TrainMeta()
    payload: Dict[str, object] = {
        "state_dict": model.cpu().state_dict(),
        "meta": asdict(meta),
        "model_class": "SmallCnn",
        "exports": {
            "notes": "Use export_and_optimize.py to produce ONNX from this checkpoint.",
        },
    }
    torch.save(payload, args.out)
    print(f"Saved checkpoint: {args.out}")
    print("meta:", json.dumps(asdict(meta), indent=2))


if __name__ == "__main__":
    main()
