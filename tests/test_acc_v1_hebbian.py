"""Tests for ACCv1Hebbian — pure Hebbian ACC variant.

The Hebbian update rule (PLAN §4.2) is:
    outer = (h_R - μ_R).T @ (h_L - μ_L) / B
    ΔW    = η · outer − λ · W
    W     ← clip(W + ΔW, −W_max, +W_max)

Tests verify each piece in isolation and the rule as a whole.
"""
from __future__ import annotations

import torch

from split_mnist.acc import ACCv1Hebbian


# -- Construction & W as buffer (not Parameter) ----------------------------------------------


def test_w_shape() -> None:
    acc = ACCv1Hebbian(hidden_dim=64)
    assert acc.W.shape == (64, 64)


def test_w_initialized_to_zero() -> None:
    acc = ACCv1Hebbian(hidden_dim=64)
    assert torch.all(acc.W == 0)


def test_w_is_buffer_not_parameter() -> None:
    """V1's W must NOT be an nn.Parameter — backprop should not touch it."""
    acc = ACCv1Hebbian(hidden_dim=64)
    # Buffer, not Parameter.
    assert not isinstance(acc.W_hebbian, torch.nn.Parameter)
    # Buffers aren't returned by .parameters().
    assert "W_hebbian" not in dict(acc.named_parameters())
    # But they ARE in state_dict (so checkpointing works).
    assert "W_hebbian" in acc.state_dict()


def test_zero_learnable_parameters() -> None:
    """V1 has no learnable parameters (Hebbian only)."""
    acc = ACCv1Hebbian(hidden_dim=64)
    n_params = sum(p.numel() for p in acc.parameters())
    assert n_params == 0


def test_buffer_follows_to_device() -> None:
    """Buffer should follow .to() (state_dict + device tracking)."""
    acc = ACCv1Hebbian(hidden_dim=64)
    # Move to a copy via .to (CPU only, but verifies the mechanism).
    acc_cpu = acc.to("cpu")
    assert acc_cpu.W_hebbian.device.type == "cpu"


# -- Forward (uses inherited ACCBase methods, but needs sanity) ------------------------------


def test_forward_shapes() -> None:
    acc = ACCv1Hebbian(hidden_dim=64)
    h_L = torch.randn(8, 64)
    h_R = torch.randn(8, 64)
    assert acc.forward_LR(h_L).shape == (8, 64)
    assert acc.forward_RL(h_R).shape == (8, 64)


def test_forward_with_zero_W_yields_zero() -> None:
    """At init W=0, forward outputs are zero."""
    acc = ACCv1Hebbian(hidden_dim=64)
    h_L = torch.randn(4, 64)
    h_R = torch.randn(4, 64)
    assert torch.all(acc.forward_LR(h_L) == 0)
    assert torch.all(acc.forward_RL(h_R) == 0)


# -- hebbian_update — the core rule ---------------------------------------------------------


def test_hebbian_update_changes_W() -> None:
    """One non-trivial batch must move W away from 0."""
    torch.manual_seed(0)
    acc = ACCv1Hebbian(hidden_dim=64, eta=0.1, decay=0.0)
    h_L = torch.randn(32, 64)
    h_R = torch.randn(32, 64)

    W_before = acc.W.clone()
    acc.hebbian_update(h_L, h_R)
    W_after = acc.W.clone()

    assert not torch.allclose(W_before, W_after)


def test_hebbian_update_no_grad_required() -> None:
    """Calling hebbian_update should not require/produce gradients."""
    acc = ACCv1Hebbian(hidden_dim=64)
    h_L = torch.randn(8, 64, requires_grad=True)
    h_R = torch.randn(8, 64, requires_grad=True)
    # Inside autograd context — should still be safe.
    acc.hebbian_update(h_L, h_R)
    # No grad created on inputs.
    assert h_L.grad is None
    assert h_R.grad is None
    # W_hebbian should have no grad attribute (it's a buffer).
    assert acc.W_hebbian.grad is None


def test_mean_centering_zeros_update_for_constant_input() -> None:
    """If every sample in the batch is identical, mean-centered update is ~0
    (regardless of decay, since W=0 at start).

    Note: float32 arithmetic introduces ~1e-14 residuals from the mean
    subtraction, so we use a tight absolute tolerance rather than equality.
    """
    acc = ACCv1Hebbian(hidden_dim=64, eta=1.0, decay=0.0)

    # All samples equal -> after subtracting the mean, centered = 0.
    fixed_L = torch.randn(64)
    fixed_R = torch.randn(64)
    h_L = fixed_L.unsqueeze(0).expand(32, -1).contiguous()
    h_R = fixed_R.unsqueeze(0).expand(32, -1).contiguous()

    acc.hebbian_update(h_L, h_R)
    # float32 noise from mean subtraction: ~1e-14 level. Use 1e-10 atol.
    assert acc.W.abs().max().item() < 1e-10


def test_hebbian_outer_product_correctness() -> None:
    """Manually compute expected ΔW and compare.

    With decay=0, W_before=0, η=1, batch B:
        ΔW = (1/B) · (h_R - μ_R).T @ (h_L - μ_L)
    """
    torch.manual_seed(7)
    D, B = 64, 16
    acc = ACCv1Hebbian(hidden_dim=D, eta=1.0, decay=0.0, w_max=1e9)  # disable clip

    h_L = torch.randn(B, D)
    h_R = torch.randn(B, D)

    expected = (h_R - h_R.mean(0, keepdim=True)).T @ (h_L - h_L.mean(0, keepdim=True)) / B

    acc.hebbian_update(h_L, h_R)
    assert torch.allclose(acc.W, expected, atol=1e-6)


def test_decay_shrinks_W_toward_zero_when_no_input_signal() -> None:
    """If we keep calling hebbian_update with constant input (zero update), then
    decay should shrink an existing W toward zero over many calls."""
    torch.manual_seed(0)
    acc = ACCv1Hebbian(hidden_dim=64, eta=0.0, decay=0.1)

    # Manually plant a non-trivial W.
    with torch.no_grad():
        acc.W_hebbian.copy_(torch.randn(64, 64) * 0.5)

    initial_norm = acc.W.norm().item()

    # Constant inputs: outer = 0; only decay drives the update.
    fixed = torch.randn(8, 64)
    for _ in range(20):
        acc.hebbian_update(fixed, fixed)

    final_norm = acc.W.norm().item()
    assert final_norm < initial_norm * 0.5, f"{initial_norm} -> {final_norm}"


def test_clipping_caps_W() -> None:
    """If we drive W with a large η for many steps, clipping should cap |W|."""
    torch.manual_seed(0)
    acc = ACCv1Hebbian(hidden_dim=64, eta=10.0, decay=0.0, w_max=0.5)

    # Fixed strongly-correlated batch so the outer product is large.
    h_L = torch.randn(32, 64)
    h_R = h_L.clone() * 3.0  # strong correlation, scaled

    for _ in range(20):
        acc.hebbian_update(h_L, h_R)

    assert acc.W.abs().max().item() <= 0.5 + 1e-6


def test_shape_mismatch_raises() -> None:
    acc = ACCv1Hebbian(hidden_dim=64)
    h_L = torch.randn(8, 64)
    h_R = torch.randn(8, 32)  # wrong dim
    try:
        acc.hebbian_update(h_L, h_R)
    except ValueError:
        return
    raise AssertionError("expected ValueError for shape mismatch")


def test_wrong_rank_raises() -> None:
    acc = ACCv1Hebbian(hidden_dim=64)
    h_L = torch.randn(64)        # 1D, not (B, D)
    h_R = torch.randn(64)
    try:
        acc.hebbian_update(h_L, h_R)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-2D input")


# -- Reconstruction loss (inherited) --------------------------------------------------------


def test_reconstruction_loss_returns_scalar() -> None:
    acc = ACCv1Hebbian(hidden_dim=64)
    h_L = torch.randn(8, 64)
    h_R = torch.randn(8, 64)
    loss = acc.reconstruction_loss(h_L, h_R)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_v1_recon_loss_has_no_autograd_path() -> None:
    """V1's W is a buffer (no requires_grad), so reconstruction_loss has
    *no autograd graph* — calling .backward() on it would raise. This is
    by design: V1 is Hebbian-only and cannot receive backprop signal.

    The combined V3 variant adds W_learned (an nn.Parameter) precisely to
    enable backprop through the same loss function.
    """
    acc = ACCv1Hebbian(hidden_dim=64)
    # Plant a known W.
    with torch.no_grad():
        acc.W_hebbian.copy_(torch.randn(64, 64) * 0.1)

    h_L = torch.randn(8, 64)
    h_R = torch.randn(8, 64)
    loss = acc.reconstruction_loss(h_L, h_R)

    # Core V1 invariant: loss has no autograd graph (W is a buffer; inputs
    # don't require grad either, so the entire computation is grad-free).
    assert not loss.requires_grad
    assert loss.grad_fn is None
    # Therefore .backward() would error; we deliberately don't call it.

    # And W_hebbian, being a buffer, has no .grad attribute populated.
    assert acc.W_hebbian.grad is None
