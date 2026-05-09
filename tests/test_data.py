"""Tests for split_mnist.data — shape, range, split sizes."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torchvision import datasets

from split_mnist.data import MNIST_MEAN, MNIST_STD, SplitMNIST, make_loaders


@pytest.fixture(scope="module")
def split_train() -> SplitMNIST:
    """Cache MNIST in-process so each test doesn't re-download."""
    return SplitMNIST(root="./data", train=True, normalize=True)


def test_split_mnist_lengths(split_train: SplitMNIST) -> None:
    assert len(split_train) == 60_000


def test_getitem_shapes(split_train: SplitMNIST) -> None:
    x_l, x_r, y = split_train[0]
    assert x_l.shape == (1, 28, 14)
    assert x_r.shape == (1, 28, 14)
    assert isinstance(y, int)
    assert 0 <= y <= 9


def test_normalization_applied(split_train: SplitMNIST) -> None:
    x_l, _, _ = split_train[0]
    # Pixels with raw value 0 become (0 - 0.1307) / 0.3081 ~ -0.4242
    # so at least one pixel must be < 0 if normalize was applied.
    assert x_l.dtype == torch.float32
    assert x_l.min() < 0


def test_left_right_no_overlap_and_concat_recovers_original() -> None:
    """Verify that concatenating left+right reconstructs the raw MNIST image."""
    ds = SplitMNIST(root="./data", train=True, normalize=False)
    raw = datasets.MNIST(root="./data", train=True, download=False, transform=None)

    x_l, x_r, _ = ds[0]
    raw_arr = np.array(raw[0][0], dtype=np.float32) / 255.0  # (28, 28) in [0, 1]
    recon = torch.cat([x_l, x_r], dim=-1).squeeze(0).numpy()
    np.testing.assert_allclose(recon, raw_arr, atol=1e-6)


def test_make_loaders_dataset_sizes() -> None:
    train_loader, val_loader, test_loader = make_loaders(
        root="./data", batch_size=128, val_size=10_000, seed=42
    )
    assert len(train_loader.dataset) == 50_000
    assert len(val_loader.dataset) == 10_000
    assert len(test_loader.dataset) == 10_000


def test_make_loaders_batch_shapes() -> None:
    train_loader, _, _ = make_loaders(
        root="./data", batch_size=64, val_size=10_000, seed=42
    )
    x_l, x_r, y = next(iter(train_loader))
    assert x_l.shape == (64, 1, 28, 14)
    assert x_r.shape == (64, 1, 28, 14)
    assert y.shape == (64,)
    assert y.dtype == torch.int64


def test_normalization_constants_match_mnist_standard() -> None:
    """Sanity check: avoid silently changing normalization."""
    assert MNIST_MEAN == pytest.approx(0.1307)
    assert MNIST_STD == pytest.approx(0.3081)
