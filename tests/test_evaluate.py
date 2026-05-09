"""Tests for split_mnist.evaluate — measurements #1~#4.

We use synthetic models (random-init, no real training) for these tests.
The point is to verify shapes, applicability matrix, and the math of
each measurement, not to test learning dynamics.
"""
from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from split_mnist.evaluate import (
    ACC_LIKE_VARIANTS,
    SPLIT_VARIANTS,
    measure_causal_coupling,
    measure_cross_activation,
    measure_position_invariance,
    measure_task_accuracy,
)
from split_mnist.train import TrainConfig, build_model


# -- Dummy data ------------------------------------------------------------------------------


class _DummySplit(Dataset):
    def __init__(self, n: int = 64, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.x_L = torch.randn(n, 1, 28, 14, generator=g)
        self.x_R = torch.randn(n, 1, 28, 14, generator=g)
        self.y = torch.randint(0, 10, (n,), generator=g)

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self.x_L[idx], self.x_R[idx], int(self.y[idx])


def _loader(n: int = 64, batch_size: int = 16, seed: int = 0) -> DataLoader:
    return DataLoader(_DummySplit(n=n, seed=seed), batch_size=batch_size, shuffle=False)


def _model(variant: str) -> dict:
    cfg = TrainConfig(variant=variant, device="cpu")
    return build_model(cfg)


# -- Measurement #1 — task accuracy --------------------------------------------------------


@pytest.mark.parametrize("variant", ["B1", "B2a", "B3", "B4", "V1", "V2", "V3"])
def test_measure_task_accuracy_runs(variant: str) -> None:
    model = _model(variant)
    acc = measure_task_accuracy(model, _loader(64, 16), variant, device="cpu")
    assert 0.0 <= acc <= 1.0


def test_measure_task_accuracy_with_random_init_is_near_chance() -> None:
    """Random-init model on 10-class MNIST: accuracy ~10%."""
    torch.manual_seed(0)
    model = _model("B3")
    acc = measure_task_accuracy(model, _loader(256, 32), "B3", device="cpu")
    # We don't assert tight bounds, just that it's in a sane range.
    assert 0.0 <= acc <= 0.5


# -- Measurement #2 applicability matrix --------------------------------------------------


def test_applicability_constants() -> None:
    """Sanity-check the ACC_LIKE / SPLIT variant tuples."""
    assert "V1" in ACC_LIKE_VARIANTS
    assert "V2" in ACC_LIKE_VARIANTS
    assert "V3" in ACC_LIKE_VARIANTS
    assert "B4" in ACC_LIKE_VARIANTS
    assert "B1" not in ACC_LIKE_VARIANTS
    assert "B2a" not in ACC_LIKE_VARIANTS
    assert "B3" not in ACC_LIKE_VARIANTS

    assert "B1" not in SPLIT_VARIANTS
    for v in ("B2a", "B3", "B4", "V1", "V2", "V3"):
        assert v in SPLIT_VARIANTS


def test_measure_cross_activation_b1_returns_nan() -> None:
    model = _model("B1")
    out = measure_cross_activation(model, _loader(), "B1", device="cpu")
    assert out["applicable_recon"] is False
    assert out["applicable_split"] is False
    assert math.isnan(out["acc_real"])
    assert math.isnan(out["acc_reconstructed"])


@pytest.mark.parametrize("variant", ["B2a", "B3"])
def test_measure_cross_activation_partial_for_no_acc_variants(variant: str) -> None:
    """Variants without an ACC component get partial results — acc_real and
    acc_ablated only; cosine and acc_reconstructed are NaN."""
    model = _model(variant)
    out = measure_cross_activation(model, _loader(), variant, device="cpu")
    assert out["applicable_split"] is True
    assert out["applicable_recon"] is False
    assert 0.0 <= out["acc_real"] <= 1.0
    assert 0.0 <= out["acc_ablated"] <= 1.0
    assert math.isnan(out["cosine"])
    assert math.isnan(out["acc_reconstructed"])


@pytest.mark.parametrize("variant", ["V1", "V2", "V3", "B4"])
def test_measure_cross_activation_full_for_acc_variants(variant: str) -> None:
    """V1/V2/V3/B4 produce the full result dict."""
    model = _model(variant)
    out = measure_cross_activation(model, _loader(), variant, device="cpu")
    assert out["applicable_split"] is True
    assert out["applicable_recon"] is True
    for k in (
        "cosine",
        "cosine_std",
        "acc_real",
        "acc_ablated",
        "acc_reconstructed",
    ):
        v = out[k]
        assert isinstance(v, float)
        assert math.isfinite(v) or k == "cosine_std", f"{k}={v}"


def test_measure_cross_activation_both_directions_runnable() -> None:
    model = _model("V3")
    for direction in ("right_ablation", "left_ablation"):
        out = measure_cross_activation(
            model, _loader(), "V3", device="cpu", direction=direction
        )
        assert out["direction"] == direction
        assert out["n_samples"] > 0


def test_measure_cross_activation_invalid_direction_raises() -> None:
    model = _model("V3")
    with pytest.raises(ValueError):
        measure_cross_activation(model, _loader(), "V3", device="cpu", direction="bogus")


def test_measure_cross_activation_with_zero_W_yields_zero_cosine() -> None:
    """At init (W=0), reconstructed ĥ_R is the zero vector → cosine
    similarity with non-zero h_R is 0 (or NaN by convention).
    PyTorch's F.cosine_similarity handles the zero case by returning 0
    (since it adds eps to the denominator)."""
    model = _model("V2")  # ACCv2Recon's W is exactly zero at init
    out = measure_cross_activation(
        model, _loader(64, 16), "V2", device="cpu"
    )
    # ĥ_R = h_L @ 0 = 0 → cos(0, h_R) ≈ 0.
    assert abs(out["cosine"]) < 0.05, f"got {out['cosine']}"


# -- Measurement #3 — causal coupling ------------------------------------------------------


def test_measure_causal_coupling_b1_returns_nan() -> None:
    model = _model("B1")
    out = measure_causal_coupling(model, _loader(), "B1", device="cpu")
    assert out["applicable"] is False
    assert all(math.isnan(a) for a in out["accuracies"])


@pytest.mark.parametrize("variant", ["B2a", "B3", "B4", "V1", "V2", "V3"])
def test_measure_causal_coupling_runs(variant: str) -> None:
    model = _model(variant)
    out = measure_causal_coupling(
        model, _loader(64, 16), variant, device="cpu",
        epsilons=(0.0, 0.5, 1.0),
    )
    assert out["applicable"] is True
    assert len(out["accuracies"]) == 3
    for a in out["accuracies"]:
        assert 0.0 <= a <= 1.0


def test_measure_causal_coupling_higher_eps_does_not_increase_acc() -> None:
    """Adding noise to h_L should *not* improve accuracy on average. We allow
    small jitter due to randomness in the noise itself.
    Random-init networks may give nearly chance-level accuracy at every ε,
    so we verify the *direction* is non-positive in expectation, not strict.
    """
    torch.manual_seed(0)
    model = _model("V3")
    out = measure_causal_coupling(
        model, _loader(256, 32), "V3", device="cpu",
        epsilons=(0.0, 1.0, 5.0),
    )
    accs = out["accuracies"]
    # Don't assert strict monotone (random init noise floor); just no big jump up.
    # Specifically: acc[2] should not be much greater than acc[0].
    assert accs[2] <= accs[0] + 0.10, f"unexpected acc growth: {accs}"


def test_measure_causal_coupling_ias_at_0p5_present_when_in_epsilons() -> None:
    model = _model("V3")
    out = measure_causal_coupling(
        model, _loader(64, 16), "V3", device="cpu",
        epsilons=(0.0, 0.5, 1.0),
    )
    # IAS = (acc[0] - acc[0.5]) / acc[0]  → finite scalar
    assert math.isfinite(out["ias_at_0p5"])


def test_measure_causal_coupling_ias_nan_when_no_0p5_in_epsilons() -> None:
    model = _model("V3")
    out = measure_causal_coupling(
        model, _loader(64, 16), "V3", device="cpu",
        epsilons=(0.0, 0.25, 1.0),
    )
    assert math.isnan(out["ias_at_0p5"])


# -- Measurement #4 — position invariance --------------------------------------------------


def test_measure_position_invariance_identical_W_gives_high_corr() -> None:
    """5 identical W's → procrustes corr ≈ 1.0."""
    torch.manual_seed(0)
    W = torch.randn(64, 64)
    out = measure_position_invariance([W.clone() for _ in range(5)])
    assert out["procrustes_corr_mean"] > 0.99
    assert out["n_seeds"] == 5
    assert len(out["procrustes_corr_per_seed"]) == 4


def test_measure_position_invariance_random_W_distinguishes_from_identical() -> None:
    """Random isotropic 64×64 matrices have a non-trivial Procrustes correlation
    baseline (~0.7-0.8) due to high-dimensional rotation flexibility — they
    are NOT near 0. The crucial property is that they are *distinguishably
    less aligned than identical W's* (which give ~1.0).

    This was measured empirically in Day 6 and led to the random-baseline
    re-calibration of measurement #4 (PLAN §5.5, §16.5 D-21).
    """
    torch.manual_seed(123)
    W_list = [torch.randn(64, 64) for _ in range(5)]
    out = measure_position_invariance(W_list)
    # Random baseline empirically ~0.74. Verify distinct from identical (~1.0).
    assert out["procrustes_corr_mean"] < 0.85, out
    # And it's not negligible — confirming the random baseline phenomenon.
    assert out["procrustes_corr_mean"] > 0.3, out


def test_measure_position_invariance_requires_at_least_two() -> None:
    with pytest.raises(ValueError):
        measure_position_invariance([torch.randn(64, 64)])


def test_measure_position_invariance_returns_correct_shape() -> None:
    """3 W's → 2 per-seed correlations."""
    torch.manual_seed(0)
    W_list = [torch.randn(64, 64) for _ in range(3)]
    out = measure_position_invariance(W_list)
    assert out["n_seeds"] == 3
    assert len(out["procrustes_corr_per_seed"]) == 2
    for r in out["procrustes_corr_per_seed"]:
        assert -1.0 <= r <= 1.0
