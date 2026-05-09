"""Tests for ACCv2Recon — pure reconstruction ACC variant."""
from __future__ import annotations

import torch

from split_mnist.acc import ACCv2Recon


# -- Construction & W shape ------------------------------------------------------------------


def test_w_shape() -> None:
    acc = ACCv2Recon(hidden_dim=64)
    assert acc.W.shape == (64, 64)


def test_w_initialized_to_zero() -> None:
    """PLAN §4.5 (1): W initialized to 0 across all variants."""
    acc = ACCv2Recon(hidden_dim=64)
    assert torch.all(acc.W == 0)


def test_w_is_nn_parameter() -> None:
    """W_learned must be an nn.Parameter so backprop reaches it."""
    acc = ACCv2Recon(hidden_dim=64)
    assert isinstance(acc.W_learned, torch.nn.Parameter)
    assert acc.W_learned.requires_grad


def test_param_count() -> None:
    """V2 has exactly hidden_dim^2 parameters (no bias, no gate)."""
    acc = ACCv2Recon(hidden_dim=64)
    n_params = sum(p.numel() for p in acc.parameters())
    assert n_params == 64 * 64


# -- Forward shapes --------------------------------------------------------------------------


def test_forward_LR_shape() -> None:
    acc = ACCv2Recon(hidden_dim=64)
    h_L = torch.randn(8, 64)
    h_R_pred = acc.forward_LR(h_L)
    assert h_R_pred.shape == (8, 64)


def test_forward_RL_shape() -> None:
    acc = ACCv2Recon(hidden_dim=64)
    h_R = torch.randn(8, 64)
    h_L_pred = acc.forward_RL(h_R)
    assert h_L_pred.shape == (8, 64)


def test_forward_with_zero_W_yields_zero() -> None:
    """At init W=0, both forward directions output zero."""
    acc = ACCv2Recon(hidden_dim=64)
    h_L = torch.randn(4, 64)
    h_R = torch.randn(4, 64)
    assert torch.all(acc.forward_LR(h_L) == 0)
    assert torch.all(acc.forward_RL(h_R) == 0)


def test_forward_LR_uses_W_transpose_correctly() -> None:
    """ĥ_R = h_L @ W.T  ⇔  ĥ_R[b, i] = Σ_j W[i, j] · h_L[b, j].

    Set W to a single 1 at (i=3, j=5) and verify only ĥ_R[:, 3] reflects h_L[:, 5].
    """
    acc = ACCv2Recon(hidden_dim=64)
    with torch.no_grad():
        acc.W_learned.zero_()
        acc.W_learned[3, 5] = 1.0
    h_L = torch.zeros(2, 64)
    h_L[:, 5] = 7.0
    h_R_pred = acc.forward_LR(h_L)
    # Only column 3 of ĥ_R should be non-zero, value = 7.0
    assert h_R_pred[0, 3].item() == 7.0
    assert h_R_pred[1, 3].item() == 7.0
    # All other columns zero.
    h_R_pred[:, 3] = 0
    assert torch.all(h_R_pred == 0)


def test_forward_RL_uses_W_correctly() -> None:
    """ĥ_L = h_R @ W  ⇔  ĥ_L[b, j] = Σ_i W[i, j] · h_R[b, i].

    With W[3, 5] = 1, set h_R[:, 3] and check h_L_pred[:, 5].
    """
    acc = ACCv2Recon(hidden_dim=64)
    with torch.no_grad():
        acc.W_learned.zero_()
        acc.W_learned[3, 5] = 1.0
    h_R = torch.zeros(2, 64)
    h_R[:, 3] = 7.0
    h_L_pred = acc.forward_RL(h_R)
    assert h_L_pred[0, 5].item() == 7.0
    assert h_L_pred[1, 5].item() == 7.0
    h_L_pred[:, 5] = 0
    assert torch.all(h_L_pred == 0)


# -- Reconstruction loss ---------------------------------------------------------------------


def test_reconstruction_loss_returns_scalar() -> None:
    acc = ACCv2Recon(hidden_dim=64)
    h_L = torch.randn(8, 64)
    h_R = torch.randn(8, 64)
    loss = acc.reconstruction_loss(h_L, h_R)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_reconstruction_loss_with_zero_W_equals_norms() -> None:
    """At init (W=0), recon loss = MSE(0, h_R) + MSE(0, h_L) = mean(h_R²) + mean(h_L²)."""
    acc = ACCv2Recon(hidden_dim=64)
    torch.manual_seed(0)
    h_L = torch.randn(16, 64)
    h_R = torch.randn(16, 64)
    loss = acc.reconstruction_loss(h_L, h_R)
    expected = (h_R**2).mean() + (h_L**2).mean()
    assert torch.allclose(loss, expected, atol=1e-6)


def test_reconstruction_loss_zero_when_W_is_identity_and_h_L_equals_h_R() -> None:
    """If h_L == h_R and W = I, then ĥ_R = h_L = h_R → loss = 0."""
    acc = ACCv2Recon(hidden_dim=64)
    with torch.no_grad():
        acc.W_learned.copy_(torch.eye(64))
    h = torch.randn(8, 64)
    loss = acc.reconstruction_loss(h, h)
    assert loss.item() < 1e-10


# -- Backprop ---------------------------------------------------------------------------------


def test_backprop_updates_W() -> None:
    """A single optimizer step on recon loss should change W away from 0."""
    torch.manual_seed(42)
    acc = ACCv2Recon(hidden_dim=64)
    opt = torch.optim.AdamW(acc.parameters(), lr=1e-2)

    h_L = torch.randn(32, 64)
    h_R = torch.randn(32, 64)

    W_before = acc.W.detach().clone()
    loss = acc.reconstruction_loss(h_L, h_R)
    loss.backward()
    opt.step()
    W_after = acc.W.detach().clone()

    assert not torch.allclose(W_before, W_after)


def test_loss_decreases_over_steps() -> None:
    """Smoke: 50 steps of optimization on the same fixed batch reduces loss > 50%."""
    torch.manual_seed(0)
    acc = ACCv2Recon(hidden_dim=64)
    opt = torch.optim.AdamW(acc.parameters(), lr=1e-2)

    h_L = torch.randn(32, 64)
    h_R = torch.randn(32, 64)

    initial = acc.reconstruction_loss(h_L, h_R).item()
    for _ in range(50):
        opt.zero_grad()
        loss = acc.reconstruction_loss(h_L, h_R)
        loss.backward()
        opt.step()
    final = acc.reconstruction_loss(h_L, h_R).item()

    assert final < initial * 0.5, f"loss did not decrease: {initial} -> {final}"


def test_grad_does_not_flow_into_inputs_when_detached() -> None:
    """When inputs are .detach()-ed (the intended PLAN §4.5 (5) usage),
    gradient should not propagate to them."""
    torch.manual_seed(0)
    acc = ACCv2Recon(hidden_dim=64)
    h_L = torch.randn(8, 64, requires_grad=True)
    h_R = torch.randn(8, 64, requires_grad=True)

    loss = acc.reconstruction_loss(h_L.detach(), h_R.detach())
    loss.backward()
    assert h_L.grad is None
    assert h_R.grad is None


# -- Hebbian no-op ---------------------------------------------------------------------------


def test_hebbian_update_is_noop_for_v2() -> None:
    """V2 has no Hebbian path; calling hebbian_update should not change W."""
    acc = ACCv2Recon(hidden_dim=64)
    with torch.no_grad():
        acc.W_learned.copy_(torch.randn(64, 64))
    W_before = acc.W.detach().clone()

    h_L = torch.randn(8, 64)
    h_R = torch.randn(8, 64)
    acc.hebbian_update(h_L, h_R)

    W_after = acc.W.detach().clone()
    assert torch.equal(W_before, W_after)
