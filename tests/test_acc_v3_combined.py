"""Tests for ACCv3Combined — the hypothesized form.

W = W_hebbian + tanh(g) · W_learned
    \\_______/   \\______________/
     buffer       Parameter, gated

Key invariants:
- W_hebbian is updated ONLY by hebbian_update (no backprop).
- W_learned, g are updated ONLY by backprop (no Hebbian path).
- At init (g=0), W = W_hebbian, so V3 behaves identically to V1.
- The two update paths must not interfere.
"""
from __future__ import annotations

import torch

from split_mnist.acc import ACCv3Combined


# -- Construction & components --------------------------------------------------------------


def test_w_shape() -> None:
    acc = ACCv3Combined(hidden_dim=64)
    assert acc.W.shape == (64, 64)


def test_w_hebbian_and_g_zero_initialized() -> None:
    """W_hebbian and g start at 0. W_learned must NOT be zero — see D-19."""
    acc = ACCv3Combined(hidden_dim=64)
    assert torch.all(acc.W_hebbian == 0)
    assert acc.g.item() == 0.0


def test_w_learned_initialized_small_random() -> None:
    """W_learned is initialized small-random (not zero) to break the
    chicken-and-egg symmetry between W_learned and g (PLAN §16.5 D-19).

    Without this, both gradients are exactly 0 and AdamW cannot recover.
    """
    acc = ACCv3Combined(hidden_dim=64)
    assert not torch.all(acc.W_learned == 0)
    # But the magnitude must be small so the residual contribution at init
    # (gated to 0 by tanh(0)) doesn't perturb anything once g starts moving.
    assert acc.W_learned.abs().max().item() < 0.1
    # And the std should be near learned_init_std default (0.01) — sanity.
    assert 1e-4 < acc.W_learned.std().item() < 0.1


def test_W_learned_init_std_is_configurable() -> None:
    """The init std is exposed as a constructor arg for ablation."""
    acc = ACCv3Combined(hidden_dim=64, learned_init_std=0.05)
    # Standard deviation of N(0, 0.05²) on a 64x64 sample should be ~0.05.
    s = acc.W_learned.std().item()
    assert 0.02 < s < 0.1, f"got std={s}"


def test_w_hebbian_is_buffer() -> None:
    acc = ACCv3Combined(hidden_dim=64)
    assert not isinstance(acc.W_hebbian, torch.nn.Parameter)
    assert "W_hebbian" not in dict(acc.named_parameters())
    assert "W_hebbian" in acc.state_dict()


def test_w_learned_is_parameter() -> None:
    acc = ACCv3Combined(hidden_dim=64)
    assert isinstance(acc.W_learned, torch.nn.Parameter)
    assert acc.W_learned.requires_grad


def test_g_is_scalar_parameter() -> None:
    acc = ACCv3Combined(hidden_dim=64)
    assert isinstance(acc.g, torch.nn.Parameter)
    assert acc.g.requires_grad
    assert acc.g.dim() == 0  # scalar


def test_param_count() -> None:
    """V3 learnable params = W_learned (D²) + g (1)."""
    acc = ACCv3Combined(hidden_dim=64)
    n_learnable = sum(p.numel() for p in acc.parameters())
    assert n_learnable == 64 * 64 + 1, f"got {n_learnable}"


# -- Effective W formula --------------------------------------------------------------------


def test_initial_W_equals_W_hebbian_via_closed_gate() -> None:
    """At init: g=0, tanh(0)=0 → tanh(g)·W_learned = 0 → W = W_hebbian.

    Note: W_learned itself is non-zero (D-19 fix) but the closed gate
    nullifies it at the first step. This preserves Flamingo's
    "no perturbation at start" property while still giving g a non-zero
    gradient signal so it can begin to open.
    """
    acc = ACCv3Combined(hidden_dim=64)
    # W_learned is small-random but the closed gate masks it.
    assert torch.allclose(acc.W, acc.W_hebbian)
    # W_hebbian is zero, so W is also exactly zero.
    assert torch.allclose(acc.W, torch.zeros_like(acc.W))


def test_W_formula_with_known_components() -> None:
    """W = W_hebbian + tanh(g) · W_learned, exact match."""
    acc = ACCv3Combined(hidden_dim=64)
    with torch.no_grad():
        acc.W_hebbian.copy_(torch.randn(64, 64) * 0.1)
        acc.W_learned.copy_(torch.randn(64, 64) * 0.1)
        acc.g.fill_(0.5)

    expected = acc.W_hebbian + torch.tanh(acc.g) * acc.W_learned
    assert torch.allclose(acc.W, expected, atol=1e-6)


def test_with_g_zero_W_learned_has_no_effect() -> None:
    """g=0 → tanh(0)=0 → W_learned does not contribute to W, regardless of its value."""
    acc = ACCv3Combined(hidden_dim=64)
    with torch.no_grad():
        acc.W_hebbian.copy_(torch.randn(64, 64))
        acc.W_learned.copy_(torch.randn(64, 64) * 100.0)  # large
        acc.g.zero_()
    assert torch.allclose(acc.W, acc.W_hebbian)


# -- Two update paths must not interfere ----------------------------------------------------


def test_hebbian_update_changes_only_W_hebbian() -> None:
    """Calling hebbian_update should change W_hebbian but NOT W_learned or g."""
    torch.manual_seed(0)
    acc = ACCv3Combined(hidden_dim=64, eta=0.1, decay=0.0)
    with torch.no_grad():
        acc.W_learned.copy_(torch.randn(64, 64) * 0.1)
        acc.g.fill_(0.3)

    W_learned_before = acc.W_learned.clone()
    g_before = acc.g.item()
    W_hebbian_before = acc.W_hebbian.clone()

    h_L = torch.randn(32, 64)
    h_R = torch.randn(32, 64)
    acc.hebbian_update(h_L, h_R)

    # W_hebbian moved.
    assert not torch.allclose(W_hebbian_before, acc.W_hebbian)
    # W_learned and g untouched.
    assert torch.equal(W_learned_before, acc.W_learned)
    assert acc.g.item() == g_before


def test_backprop_changes_only_W_learned_and_g() -> None:
    """A gradient step on recon_loss should update W_learned and g, NOT W_hebbian."""
    torch.manual_seed(0)
    acc = ACCv3Combined(hidden_dim=64)
    # Plant non-zero W_hebbian so we can verify it's NOT modified by backprop.
    with torch.no_grad():
        acc.W_hebbian.copy_(torch.randn(64, 64) * 0.1)
        acc.g.fill_(0.3)  # non-zero so the gate is open

    W_hebbian_before = acc.W_hebbian.clone()
    W_learned_before = acc.W_learned.clone()
    g_before = acc.g.item()

    opt = torch.optim.AdamW(acc.parameters(), lr=1e-2)
    h_L = torch.randn(16, 64)
    h_R = torch.randn(16, 64)

    loss = acc.reconstruction_loss(h_L, h_R)
    loss.backward()
    opt.step()

    # W_hebbian (buffer) untouched.
    assert torch.equal(W_hebbian_before, acc.W_hebbian)
    # W_learned moved.
    assert not torch.allclose(W_learned_before, acc.W_learned)
    # g moved (the gate is being learned too).
    assert acc.g.item() != g_before


def test_g_has_grad_after_backward() -> None:
    """The gate must receive a gradient signal."""
    torch.manual_seed(0)
    acc = ACCv3Combined(hidden_dim=64)
    # Open gate slightly so W_learned contributes (otherwise tanh'(0)=1 still passes
    # gradient, but make signal non-trivial via small W_learned).
    with torch.no_grad():
        acc.W_learned.copy_(torch.randn(64, 64) * 0.1)

    h_L = torch.randn(8, 64)
    h_R = torch.randn(8, 64)
    loss = acc.reconstruction_loss(h_L, h_R)
    loss.backward()

    assert acc.g.grad is not None
    assert torch.isfinite(acc.g.grad)


def test_w_hebbian_has_no_grad() -> None:
    """W_hebbian is a buffer — `.grad` must remain None after backward."""
    acc = ACCv3Combined(hidden_dim=64)
    h_L = torch.randn(8, 64)
    h_R = torch.randn(8, 64)
    loss = acc.reconstruction_loss(h_L, h_R)
    loss.backward()
    assert acc.W_hebbian.grad is None


# -- Combined behavior: V3 = V1 at g=0, plus learnable refinement ---------------------------


def test_v3_equals_v1_when_g_is_zero_at_init() -> None:
    """With g=0, V3's effective W is purely W_hebbian (the gate masks
    W_learned no matter how big it is).

    Note: this is *only* true while g hasn't yet moved. Once g is updated
    by backprop, V3 diverges from V1.
    """
    from split_mnist.acc import ACCv1Hebbian

    torch.manual_seed(0)
    v1 = ACCv1Hebbian(hidden_dim=64, eta=0.1, decay=0.0)
    v3 = ACCv3Combined(hidden_dim=64, eta=0.1, decay=0.0)

    h_L = torch.randn(32, 64)
    h_R = torch.randn(32, 64)

    # Apply same Hebbian update to both.
    v1.hebbian_update(h_L, h_R)
    v3.hebbian_update(h_L, h_R)

    # With v3.g still at 0, tanh(0)·W_learned = 0 — gate is closed.
    # Effective W must equal v1.W, regardless of v3.W_learned's value.
    assert torch.allclose(v1.W, v3.W, atol=1e-6)

    # And recon losses on a fresh probe must be identical.
    h_L_probe = torch.randn(8, 64)
    h_R_probe = torch.randn(8, 64)
    loss_v1 = v1.reconstruction_loss(h_L_probe, h_R_probe)
    loss_v3 = v3.reconstruction_loss(h_L_probe, h_R_probe)
    assert torch.allclose(loss_v1, loss_v3, atol=1e-6)


def test_combined_optimization_smoke() -> None:
    """Smoke: V3 trained jointly (backprop + Hebbian) on a learnable mapping
    should reduce recon loss on a held-out probe.

    Setup: every step uses a *fresh random batch* (mimicking real training
    where each minibatch is different) but with a fixed L→R relationship
    h_R = h_L + small noise. ACC should learn ~identity mapping.

    Note: a previous version of this test reused the same batch every step,
    which causes Hebbian to over-fit that single outer product (driving
    W_hebbian to its w_max clip) and conflicts with backprop. That's not
    realistic — real training sees a different batch per step.
    """
    torch.manual_seed(0)
    acc = ACCv3Combined(hidden_dim=64, eta=0.05, decay=0.001)
    opt = torch.optim.AdamW(
        [p for p in acc.parameters() if p.requires_grad],
        lr=1e-2,
    )

    # Held-out probe: never seen during training. h_R = h_L + small noise.
    probe_L = torch.randn(64, 64)
    probe_R = probe_L + torch.randn(64, 64) * 0.1

    initial = acc.reconstruction_loss(probe_L, probe_R).item()

    # Train on 100 *different* random batches that share the same mapping.
    for _ in range(100):
        h_L = torch.randn(32, 64)
        h_R = h_L + torch.randn(32, 64) * 0.1  # same L->R mapping as probe
        opt.zero_grad()
        loss = acc.reconstruction_loss(h_L, h_R)
        loss.backward()
        opt.step()
        acc.hebbian_update(h_L.detach(), h_R.detach())

    final = acc.reconstruction_loss(probe_L, probe_R).item()
    assert final < initial * 0.5, f"loss did not decrease enough: {initial} -> {final}"
