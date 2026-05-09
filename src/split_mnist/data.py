"""SplitMNIST dataset and DataLoader builders.

Each MNIST image is split into a left half (cols 0-13) and a right half
(cols 14-27), simulating the bilateral input that left/right "hemispheres"
receive in the split-brain analogy.

See PLAN §3.2 (1) and §7.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


# MNIST standard normalization constants.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class SplitMNIST(Dataset):
    """Wrap MNIST and yield (x_left, x_right, y) per item.

    Each MNIST 28x28 image is sliced down the middle:
      - x_left:  cols [0, 14) -> shape (1, 28, 14)
      - x_right: cols [14, 28) -> shape (1, 28, 14)
    No overlap. See PLAN §3.2 decision (1).
    """

    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        normalize: bool = True,
    ) -> None:
        tx: list = [transforms.ToTensor()]
        if normalize:
            tx.append(transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)))
        self._mnist = datasets.MNIST(
            root=str(Path(root).expanduser()),
            train=train,
            download=True,
            transform=transforms.Compose(tx),
        )

    def __len__(self) -> int:
        return len(self._mnist)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, int]:
        x, y = self._mnist[idx]  # x: (1, 28, 28), y: int
        x_left = x[:, :, :14].contiguous()  # (1, 28, 14)
        x_right = x[:, :, 14:].contiguous()  # (1, 28, 14)
        return x_left, x_right, int(y)


def make_loaders(
    root: str = "./data",
    batch_size: int = 128,
    val_size: int = 10_000,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders.

    Train/val split: 60k -> (60k - val_size) / val_size, deterministic in `seed`.
    Test: full 10k MNIST test set.

    See PLAN §7.
    """
    full_train = SplitMNIST(root=root, train=True, normalize=True)
    test = SplitMNIST(root=root, train=False, normalize=True)

    n_total = len(full_train)
    n_train = n_total - val_size
    train_set, val_set = random_split(
        full_train,
        [n_train, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader
