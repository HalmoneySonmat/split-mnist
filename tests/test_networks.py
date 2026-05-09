"""Tests for split_mnist.networks — forward shapes, parameter counts."""
from __future__ import annotations

import torch

from split_mnist.networks import (
    Classifier,
    HalfCNN,
    IndependentClassifiers,
    SingleCNN,
)


def test_half_cnn_shape() -> None:
    net = HalfCNN(hidden_dim=64)
    x = torch.randn(8, 1, 28, 14)
    out = net(x)
    assert out.shape == (8, 64)


def test_half_cnn_independent_weights_with_different_seeds() -> None:
    """Two HalfCNN instances built under different torch seeds should have different weights."""
    torch.manual_seed(42)
    left = HalfCNN(hidden_dim=64)
    torch.manual_seed(43)
    right = HalfCNN(hidden_dim=64)
    assert not torch.allclose(left.conv1.weight, right.conv1.weight)


def test_classifier_shape() -> None:
    cls = Classifier(hidden_dim=64, n_classes=10)
    h_l = torch.randn(8, 64)
    h_r = torch.randn(8, 64)
    logits = cls(h_l, h_r)
    assert logits.shape == (8, 10)


def test_single_cnn_shape() -> None:
    net = SingleCNN(hidden_dim=64, n_classes=10)
    x = torch.randn(8, 1, 28, 28)
    logits = net(x)
    assert logits.shape == (8, 10)


def test_param_counts() -> None:
    """PLAN §8 estimate (revised after exact calculation):
        HalfCNN     ~ 48k params (conv 4.8k + fc 43k)
        Classifier  ~ 9k  params
        SingleCNN   ~ 106k params
    """
    half = HalfCNN(hidden_dim=64)
    cls = Classifier(hidden_dim=64, n_classes=10)
    single = SingleCNN(hidden_dim=64, n_classes=10)

    half_params = sum(p.numel() for p in half.parameters())
    cls_params = sum(p.numel() for p in cls.parameters())
    single_params = sum(p.numel() for p in single.parameters())

    # Loose bounds — but tight enough to catch architectural regressions.
    assert 40_000 < half_params < 60_000, f"HalfCNN params={half_params}"
    assert 5_000 < cls_params < 15_000, f"Classifier params={cls_params}"
    assert 80_000 < single_params < 130_000, f"SingleCNN params={single_params}"


def test_forward_no_nan() -> None:
    """Smoke: random input -> finite output."""
    half = HalfCNN(hidden_dim=64).eval()
    cls = Classifier(hidden_dim=64, n_classes=10).eval()
    single = SingleCNN(hidden_dim=64, n_classes=10).eval()

    x_half = torch.randn(2, 1, 28, 14)
    x_full = torch.randn(2, 1, 28, 28)
    h_l = half(x_half)
    h_r = half(x_half)  # same arch, different forward
    assert torch.isfinite(h_l).all()
    assert torch.isfinite(cls(h_l, h_r)).all()
    assert torch.isfinite(single(x_full)).all()


def test_half_cnn_relu_outputs_nonnegative() -> None:
    """The final ReLU should make hidden_L >= 0 elementwise."""
    net = HalfCNN(hidden_dim=64).eval()
    x = torch.randn(4, 1, 28, 14)
    out = net(x)
    assert (out >= 0).all()


def test_single_cnn_default_hidden_unused_in_logits_shape() -> None:
    """Changing hidden_dim should still yield (B, n_classes) logits."""
    net = SingleCNN(hidden_dim=128, n_classes=10).eval()
    x = torch.randn(2, 1, 28, 28)
    out = net(x)
    assert out.shape == (2, 10)


# -- IndependentClassifiers (B2 baseline) --------------------------------------------------


def test_independent_classifiers_shapes() -> None:
    cls = IndependentClassifiers(hidden_dim=64, n_classes=10)
    h_L = torch.randn(8, 64)
    h_R = torch.randn(8, 64)
    logits_L, logits_R = cls(h_L, h_R)
    assert logits_L.shape == (8, 10)
    assert logits_R.shape == (8, 10)


def test_independent_classifiers_left_right_truly_independent() -> None:
    """Modifying h_R must not affect logits_L (and vice versa)."""
    cls = IndependentClassifiers(hidden_dim=64, n_classes=10).eval()
    h_L = torch.randn(4, 64)
    h_R1 = torch.randn(4, 64)
    h_R2 = torch.randn(4, 64)

    logits_L_a, _ = cls(h_L, h_R1)
    logits_L_b, _ = cls(h_L, h_R2)
    # Same h_L, different h_R → logits_L unchanged.
    assert torch.equal(logits_L_a, logits_L_b)


def test_independent_classifiers_avg_logits() -> None:
    cls = IndependentClassifiers(hidden_dim=64, n_classes=10).eval()
    h_L = torch.randn(4, 64)
    h_R = torch.randn(4, 64)

    logits_L, logits_R = cls(h_L, h_R)
    avg = cls.avg_logits(h_L, h_R)
    assert avg.shape == (4, 10)
    assert torch.allclose(avg, (logits_L + logits_R) / 2)


def test_independent_classifiers_param_count() -> None:
    """IndependentClassifiers has roughly 2x the param count of one Classifier
    (since it has two parallel two-stage MLPs)."""
    cls = Classifier(hidden_dim=64, n_classes=10)
    indep = IndependentClassifiers(hidden_dim=64, n_classes=10)

    n_cls = sum(p.numel() for p in cls.parameters())
    n_indep = sum(p.numel() for p in indep.parameters())

    # IndependentClassifiers has two heads of (64->64)+(64->10) each.
    # Classifier has one (128->64)+(64->10) head.
    # The numbers aren't 2x exactly but should be in the same ballpark.
    # Hard ceiling: < 2.5x to catch architectural regressions.
    assert n_indep > n_cls
    assert n_indep < 2.5 * n_cls, f"indep={n_indep}, cls={n_cls}"


def test_independent_classifiers_finite_outputs() -> None:
    cls = IndependentClassifiers(hidden_dim=64, n_classes=10).eval()
    h_L = torch.randn(2, 64)
    h_R = torch.randn(2, 64)
    logits_L, logits_R = cls(h_L, h_R)
    assert torch.isfinite(logits_L).all()
    assert torch.isfinite(logits_R).all()
