"""Smoke tests for train.py — verify all 6 variants run end-to-end on tiny
synthetic data. The MNIST 1-epoch smoke is a separate manual run (Day 4c).

We keep these tests fast (<10s total) by:
- using a tiny in-memory dataset (64 samples)
- 1 epoch, batch_size=8
- early-stop patience set high so the first epoch always counts
"""
from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from split_mnist.train import (
    ACC_VARIANTS,
    HEBBIAN_VARIANTS,
    RECON_VARIANTS,
    VALID_VARIANTS,
    TrainConfig,
    build_model,
    set_seed,
    train_one_run,
)


# -- Dummy data ------------------------------------------------------------------------------


class _DummySplitMNIST(Dataset):
    """In-memory random split-MNIST proxy. Same shape contract as SplitMNIST."""

    def __init__(self, n: int = 64, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.x_L = torch.randn(n, 1, 28, 14, generator=g)
        self.x_R = torch.randn(n, 1, 28, 14, generator=g)
        self.y = torch.randint(0, 10, (n,), generator=g)

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self.x_L[idx], self.x_R[idx], int(self.y[idx])


def _dummy_loaders(n: int = 64, batch_size: int = 8) -> tuple:
    ds = _DummySplitMNIST(n=n)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader, loader, loader


def _smoke_cfg(variant: str, **overrides) -> TrainConfig:
    """Tiny config that completes 1 epoch in <2 seconds."""
    base = dict(
        variant=variant,
        n_epochs=1,
        batch_size=8,
        early_stop_patience=99,  # don't early-stop in 1 epoch
        log_every=999,           # silence per-step logging in tests
        device="cpu",
    )
    base.update(overrides)
    return TrainConfig(**base)


# -- Config validation ----------------------------------------------------------------------


def test_invalid_variant_raises() -> None:
    with pytest.raises(ValueError):
        TrainConfig(variant="V99")


def test_valid_variants_constants_consistent() -> None:
    """RECON_VARIANTS and HEBBIAN_VARIANTS must be subsets of VALID_VARIANTS."""
    assert set(ACC_VARIANTS).issubset(VALID_VARIANTS)
    assert set(RECON_VARIANTS).issubset(VALID_VARIANTS)
    assert set(HEBBIAN_VARIANTS).issubset(VALID_VARIANTS)


# -- build_model: dict keys per variant -----------------------------------------------------


def test_build_model_b1() -> None:
    cfg = _smoke_cfg("B1")
    m = build_model(cfg)
    assert set(m.keys()) == {"single_cnn"}


def test_build_model_b3() -> None:
    cfg = _smoke_cfg("B3")
    m = build_model(cfg)
    assert set(m.keys()) == {"left", "right", "classifier"}


def test_build_model_b4() -> None:
    cfg = _smoke_cfg("B4")
    m = build_model(cfg)
    assert set(m.keys()) == {"left", "right", "classifier", "adapter"}


def test_build_model_v_variants() -> None:
    for v in ("V1", "V2", "V3"):
        cfg = _smoke_cfg(v)
        m = build_model(cfg)
        assert set(m.keys()) == {"left", "right", "classifier", "acc"}


def test_v3_has_more_params_than_v2_and_v1() -> None:
    """V3 = V1 + V2 worth of param structure, so its learnable param count
    should be the largest of the three (V2's W_learned + V3's extra g)."""
    n_params = {}
    for v in ("V1", "V2", "V3"):
        cfg = _smoke_cfg(v)
        m = build_model(cfg)
        n_params[v] = sum(
            p.numel() for mm in m.values() for p in mm.parameters() if p.requires_grad
        )
    # V1 has no learnable params on the ACC; V2 has W_learned; V3 has W_learned + g.
    assert n_params["V3"] == n_params["V2"] + 1, n_params  # +1 for g
    assert n_params["V2"] > n_params["V1"]  # V1's ACC has 0 learnable


# -- Seed reproducibility ------------------------------------------------------------------


def test_set_seed_makes_initialization_reproducible() -> None:
    set_seed(42)
    cfg = _smoke_cfg("V3")
    m1 = build_model(cfg)
    set_seed(42)
    m2 = build_model(cfg)
    # Compare the left CNN's first conv weight as a proxy.
    assert torch.equal(m1["left"].conv1.weight, m2["left"].conv1.weight)


# -- Smoke run: every variant trains for 1 epoch -------------------------------------------


@pytest.mark.parametrize("variant", VALID_VARIANTS)
def test_smoke_run_completes(variant: str) -> None:
    """Run 1 epoch on dummy data for every variant. Verify the result schema."""
    cfg = _smoke_cfg(variant)
    loaders = _dummy_loaders(n=64, batch_size=8)

    result = train_one_run(cfg, _data_loaders=loaders)

    # Required keys.
    for k in ("val_acc_curve", "best_val_acc", "best_test_acc", "final_metrics"):
        assert k in result, f"missing key {k!r} for variant {variant}"

    # 1 epoch ran.
    assert len(result["val_acc_curve"]) == 1
    # val_acc is a real number in [0, 1].
    assert 0.0 <= result["best_val_acc"] <= 1.0
    assert 0.0 <= result["best_test_acc"] <= 1.0


@pytest.mark.parametrize("variant", VALID_VARIANTS)
def test_smoke_run_loss_finite(variant: str) -> None:
    """Final-step losses must be finite for every variant."""
    cfg = _smoke_cfg(variant)
    loaders = _dummy_loaders(n=64, batch_size=8)

    result = train_one_run(cfg, _data_loaders=loaders)

    fm = result["final_metrics"]
    assert "loss" in fm
    assert torch.isfinite(torch.tensor(fm["loss"])), fm
    if variant in RECON_VARIANTS:
        assert "loss_recon" in fm
        assert torch.isfinite(torch.tensor(fm["loss_recon"])), fm


# -- V1's classification loss must be backproppable even though ACC isn't ------------------


def test_v1_step_does_not_error_despite_buffer_W() -> None:
    """V1's W is a buffer, but classification loss path doesn't touch ACC,
    so backward must succeed normally."""
    cfg = _smoke_cfg("V1")
    loaders = _dummy_loaders(n=24, batch_size=8)
    result = train_one_run(cfg, _data_loaders=loaders)
    # If we got here, backward worked.
    assert result["best_val_acc"] >= 0.0


# -- V3 records g for diagnostics (Day 4c chicken-and-egg check) ---------------------------


def test_v3_step_records_g() -> None:
    """V3's _step should record g in final_metrics so we can monitor symmetry-breaking."""
    cfg = _smoke_cfg("V3")
    loaders = _dummy_loaders(n=24, batch_size=8)
    result = train_one_run(cfg, _data_loaders=loaders)
    assert "g" in result["final_metrics"]
    # At init g=0; after a few steps it may or may not have moved (D-19 issue).
    # We only assert presence, not value.
