"""Tests for CrossAttnAdapter (B4 baseline).

B4 is intentionally identical in W-shape and capacity to ACCv2Recon.
The differences from ACC are:
- forward returns (h_L', h_R') instead of separate forward_LR/forward_RL.
- W is trained ONLY by backprop on classification loss (no recon, no Hebbian).
- Always applies a bidirectional residual update.

These tests verify the linear-bridge formula, not attention semantics.
"""
from __future__ import annotations

import torch

from split_mnist.acc import ACCv2Recon, CrossAttnAdapter


# -- Construction & W ------------------------------------------------------------------------


def test_w_shape() -> None:
    adapter = CrossAttnAdapter(hidden_dim=64)
    assert adapter.W.shape == (64, 64)


def test_w_is_parameter() -> None:
    adapter = CrossAttnAdapter(hidden_dim=64)
    assert isinstance(adapter.W, torch.nn.Parameter)
    assert adapter.W.requires_grad


def test_w_initialized_to_zero() -> None:
    """At init, adapter is identity (no cross-influence)."""
    adapter = CrossAttnAdapter(hidden_dim=64)
    assert torch.all(adapter.W == 0)


def test_param_count_matches_ACCv2Recon() -> None:
    """B4's W must have the same parameter count as ACCv2Recon's W —
    capacity-matched ablation. The only difference is the learning signal."""
    adapter = CrossAttnAdapter(hidden_dim=64)
    acc = ACCv2Recon(hidden_dim=64)

    n_b4 = sum(p.numel() for p in adapter.parameters())
    n_acc = sum(p.numel() for p in acc.parameters())
    assert n_b4 == n_acc == 64 * 64


# -- Forward shape & residual identity at init ----------------------------------------------


def test_forward_returns_tuple_of_correct_shapes() -> None:
    adapter = CrossAttnAdapter(hidden_dim=64)
    h_L = torch.randn(8, 64)
    h_R = torch.randn(8, 64)
    h_L_prime, h_R_prime = adapter(h_L, h_R)
    assert h_L_prime.shape == (8, 64)
    assert h_R_prime.shape == (8, 64)


def test_forward_at_init_is_identity() -> None:
    """W=0 → h_L' = h_L, h_R' = h_R. Adapter must not perturb at start."""
    adapter = CrossAttnAdapter(hidden_dim=64)
    h_L = torch.randn(4, 64)
    h_R = torch.randn(4, 64)
    h_L_prime, h_R_prime = adapter(h_L, h_R)
    assert torch.equal(h_L_prime, h_L)
    assert torch.equal(h_R_prime, h_R)


# -- Linear bridge formula --------------------------------------------------------------------


def test_h_L_prime_uses_W_to_pull_from_h_R() -> None:
    """With h_L = 0 and W[i, j]=1 at one entry, h_L_prime should equal h_R @ W.

    h_L_prime = 0 + h_R @ W → check that only the j-th column of h_L_prime
    fires when only the i-th column of h_R is non-zero.
    """
    adapter = CrossAttnAdapter(hidden_dim=64)
    with torch.no_grad():
        adapter.W.zero_()
        adapter.W[3, 5] = 1.0   # h_R[:, 3] flows into h_L_prime[:, 5]

    h_L = torch.zeros(2, 64)
    h_R = torch.zeros(2, 64)
    h_R[:, 3] = 7.0

    h_L_prime, h_R_prime = adapter(h_L, h_R)

    # h_L_prime[:, 5] should be 7.0 (h_R[:, 3] * W[3, 5]).
    assert h_L_prime[0, 5].item() == 7.0
    assert h_L_prime[1, 5].item() == 7.0
    # All other columns of h_L_prime are 0 (since h_L=0 and only W[3,5]=1).
    h_L_prime[:, 5] = 0
    assert torch.all(h_L_prime == 0)
    # h_R_prime = h_R + 0 @ W.T = h_R (h_L is 0).
    assert torch.equal(h_R_prime, h_R)


def test_h_R_prime_uses_W_T_to_pull_from_h_L() -> None:
    """Symmetric direction: h_R_prime = h_R + h_L @ W.T.

    With h_R=0, W[3, 5]=1, and h_L[:, 5] non-zero, expect h_R_prime[:, 3] to fire.
    """
    adapter = CrossAttnAdapter(hidden_dim=64)
    with torch.no_grad():
        adapter.W.zero_()
        adapter.W[3, 5] = 1.0

    h_L = torch.zeros(2, 64)
    h_R = torch.zeros(2, 64)
    h_L[:, 5] = 7.0

    h_L_prime, h_R_prime = adapter(h_L, h_R)

    # h_R_prime[:, 3] = h_R[:, 3] + h_L[:, 5] * (W.T)[5, 3] = 0 + 7.0 * W[3, 5] = 7.0
    assert h_R_prime[0, 3].item() == 7.0
    assert h_R_prime[1, 3].item() == 7.0
    h_R_prime[:, 3] = 0
    assert torch.all(h_R_prime == 0)


def test_uses_same_W_in_both_directions() -> None:
    """Confirm the implementation uses one shared W (not W_LR and W_RL separately)."""
    adapter = CrossAttnAdapter(hidden_dim=64)
    # If only one nn.Parameter exists, this passes; otherwise the count differs.
    params = list(adapter.parameters())
    assert len(params) == 1
    assert params[0] is adapter.W


# -- Residual structure -----------------------------------------------------------------------


def test_residual_difference_equals_h_R_at_W() -> None:
    """h_L_prime − h_L = h_R @ W (residual structure)."""
    adapter = CrossAttnAdapter(hidden_dim=64)
    with torch.no_grad():
        adapter.W.copy_(torch.randn(64, 64) * 0.1)
    h_L = torch.randn(8, 64)
    h_R = torch.randn(8, 64)
    h_L_prime, h_R_prime = adapter(h_L, h_R)

    expected_dL = h_R @ adapter.W
    expected_dR = h_L @ adapter.W.T
    assert torch.allclose(h_L_prime - h_L, expected_dL, atol=1e-6)
    assert torch.allclose(h_R_prime - h_R, expected_dR, atol=1e-6)


# -- Backprop ---------------------------------------------------------------------------------


def test_backprop_into_W_through_classification_proxy() -> None:
    """Simulate training: build a tiny classifier on top of the adapter and
    verify that classification loss updates W."""
    torch.manual_seed(0)
    adapter = CrossAttnAdapter(hidden_dim=64)
    classifier = torch.nn.Linear(128, 10)  # mimics Classifier's first stage

    opt = torch.optim.AdamW(
        list(adapter.parameters()) + list(classifier.parameters()),
        lr=1e-2,
    )

    h_L = torch.randn(16, 64)
    h_R = torch.randn(16, 64)
    y = torch.randint(0, 10, (16,))

    W_before = adapter.W.detach().clone()

    h_L_prime, h_R_prime = adapter(h_L, h_R)
    logits = classifier(torch.cat([h_L_prime, h_R_prime], dim=-1))
    loss = torch.nn.functional.cross_entropy(logits, y)

    loss.backward()
    opt.step()

    W_after = adapter.W.detach().clone()
    assert not torch.allclose(W_before, W_after), "W did not update from classification loss"


def test_grad_flows_into_inputs() -> None:
    """Unlike ACC's separated learning, B4's adapter is in the *main* autograd
    path — gradients should flow back into the input hiddens (and thus to the
    CNN backbones in real training)."""
    torch.manual_seed(0)
    adapter = CrossAttnAdapter(hidden_dim=64)
    h_L = torch.randn(8, 64, requires_grad=True)
    h_R = torch.randn(8, 64, requires_grad=True)

    h_L_prime, h_R_prime = adapter(h_L, h_R)
    target_loss = (h_L_prime + h_R_prime).pow(2).mean()
    target_loss.backward()

    assert h_L.grad is not None
    assert h_R.grad is not None


def test_no_hebbian_update_method() -> None:
    """B4 is purely backprop-trained. Sanity: it should not have or need a
    hebbian_update method."""
    adapter = CrossAttnAdapter(hidden_dim=64)
    assert not hasattr(adapter, "hebbian_update")


def test_finite_outputs() -> None:
    adapter = CrossAttnAdapter(hidden_dim=64)
    with torch.no_grad():
        adapter.W.copy_(torch.randn(64, 64))
    h_L = torch.randn(4, 64)
    h_R = torch.randn(4, 64)
    h_L_prime, h_R_prime = adapter(h_L, h_R)
    assert torch.isfinite(h_L_prime).all()
    assert torch.isfinite(h_R_prime).all()
