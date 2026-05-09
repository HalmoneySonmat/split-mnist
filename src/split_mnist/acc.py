"""Artificial Corpus Callosum (ACC) — bidirectional mapping between hidden_L and hidden_R.

W shape convention:
    W ∈ ℝ^(D, D), where D = hidden_dim.
    W[i, j] = coactivation strength between left unit j and right unit i.

Forward equations (PLAN §3.2 (5), §4):
    ĥ_R = W · h_L         (batched: hidden_L @ W.T)
    ĥ_L = W^T · h_R       (batched: hidden_R @ W)

Variants (PLAN §4.1):
    ACCv1Hebbian  — pure Hebbian update, no backprop          (this file)
    ACCv2Recon    — pure reconstruction loss, backprop only   (this file)
    ACCv3Combined — W = W_hebbian + tanh(g) · W_learned       (this file)
    CrossAttnAdapter — B4 baseline, joint-trained linear bridge (this file)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ACCBase(nn.Module):
    """Abstract base for ACC variants.

    Subclasses must:
    - expose an effective W matrix via the `W` property (shape (D, D)),
    - optionally override `hebbian_update` (default no-op).

    Reconstruction loss (`reconstruction_loss`) is shared by all variants
    and computed from `forward_LR` / `forward_RL`.
    """

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

    @property
    def W(self) -> Tensor:
        """The effective W ∈ ℝ^(D, D) used for forward. Subclasses define."""
        raise NotImplementedError

    def forward_LR(self, hidden_L: Tensor) -> Tensor:
        """Predict ĥ_R from h_L.

        Args:
            hidden_L: (B, D)
        Returns:
            ĥ_R: (B, D), equal to hidden_L @ W.T (i.e. W · h_L for column vectors).
        """
        return hidden_L @ self.W.T

    def forward_RL(self, hidden_R: Tensor) -> Tensor:
        """Predict ĥ_L from h_R.

        Args:
            hidden_R: (B, D)
        Returns:
            ĥ_L: (B, D), equal to hidden_R @ W (i.e. W^T · h_R for column vectors).
        """
        return hidden_R @ self.W

    def reconstruction_loss(
        self, hidden_L: Tensor, hidden_R: Tensor
    ) -> Tensor:
        """Bidirectional MSE between predicted and true hidden states.

        L_recon = ‖ĥ_R − h_R‖² + ‖ĥ_L − h_L‖²

        Note: callers should pass detached hiddens (PLAN §4.5 (5)) so that
        ACC training does not propagate gradients into the CNN backbones.

        Returns:
            Scalar tensor.
        """
        h_R_pred = self.forward_LR(hidden_L)
        h_L_pred = self.forward_RL(hidden_R)
        return F.mse_loss(h_R_pred, hidden_R) + F.mse_loss(h_L_pred, hidden_L)

    def hebbian_update(self, hidden_L: Tensor, hidden_R: Tensor) -> None:
        """In-place update of any Hebbian-tracked weights. No-op for variants
        without Hebbian learning (e.g. V2). Implementations must use no_grad.
        """
        return None


class ACCv2Recon(ACCBase):
    """V2: Pure reconstruction.

    W = W_learned (single nn.Parameter), updated by backprop on
    `reconstruction_loss` only. No Hebbian.

    Initialization: W = 0 (PLAN §4.5 (1)). At step 0 the loss equals
    ‖h_R‖² + ‖h_L‖² (mean-squared norms), so the gradient signal is
    immediately non-trivial.
    """

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__(hidden_dim)
        self.W_learned = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))

    @property
    def W(self) -> Tensor:
        return self.W_learned

    # hebbian_update inherits no-op from ACCBase.


class ACCv1Hebbian(ACCBase):
    """V1: Pure Hebbian.

    W = W_hebbian, a non-learnable buffer updated by
    `hebbian_update` only. No backprop touches it.

    Update rule (PLAN §4.2, mean-centered Hebbian, batch averaged):
        μ_L = mean(h_L over batch)
        μ_R = mean(h_R over batch)
        outer = (h_R - μ_R).T @ (h_L - μ_L) / B    # shape (D_R, D_L)
        ΔW = η · outer − λ · W
        W ← clip(W + ΔW, −W_max, +W_max)

    Initialization: W_hebbian = 0 (PLAN §4.5 (1)).

    Note on parameter count: this module has *zero* nn.Parameters. The
    weight is a buffer, so it is checkpointed (state_dict) and follows
    `.to(device)` calls, but autograd does not see it.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        eta: float = 0.01,
        decay: float = 0.001,
        w_max: float = 1.0,
    ) -> None:
        super().__init__(hidden_dim)
        self.eta = float(eta)
        self.decay = float(decay)
        self.w_max = float(w_max)
        # register_buffer: included in state_dict, follows .to(device),
        # but never receives backprop gradients.
        self.register_buffer("W_hebbian", torch.zeros(hidden_dim, hidden_dim))

    @property
    def W(self) -> Tensor:
        return self.W_hebbian

    @torch.no_grad()
    def hebbian_update(self, hidden_L: Tensor, hidden_R: Tensor) -> None:
        """In-place Hebbian update of W_hebbian.

        Inputs are typically already detached at the call site (PLAN §4.5 (5)).
        We additionally guard with `torch.no_grad()` so this method is safe
        to call inside an autograd context regardless.

        Args:
            hidden_L: (B, D)
            hidden_R: (B, D)
        """
        if hidden_L.shape != hidden_R.shape:
            raise ValueError(
                f"hidden_L and hidden_R must have the same shape, "
                f"got {hidden_L.shape} vs {hidden_R.shape}"
            )
        if hidden_L.dim() != 2 or hidden_L.shape[1] != self.hidden_dim:
            raise ValueError(
                f"expected hidden of shape (B, {self.hidden_dim}), "
                f"got {hidden_L.shape}"
            )

        batch_size = hidden_L.shape[0]
        mu_L = hidden_L.mean(dim=0, keepdim=True)  # (1, D)
        mu_R = hidden_R.mean(dim=0, keepdim=True)  # (1, D)
        centered_L = hidden_L - mu_L               # (B, D)
        centered_R = hidden_R - mu_R               # (B, D)

        # Outer product summed across batch, then averaged.
        # Shape: (D_R, D_L). W[i, j] tracks coactivation of right-i with left-j.
        outer = centered_R.T @ centered_L / batch_size  # (D, D)

        delta = self.eta * outer - self.decay * self.W_hebbian
        self.W_hebbian.add_(delta)
        self.W_hebbian.clamp_(-self.w_max, self.w_max)


class ACCv3Combined(ACCv1Hebbian):
    """V3: Hebbian + reconstruction (the hypothesized form).

    W = W_hebbian + tanh(g) · W_learned
        \\_______/   \\______________/
         Hebbian      backprop, gated

    - W_hebbian: non-learnable buffer, updated only by `hebbian_update`
                 (inherited from ACCv1Hebbian).
    - W_learned: nn.Parameter, updated by backprop on `reconstruction_loss`.
    - g: scalar nn.Parameter, learnable. Init to 0 → tanh(0) = 0 → at
         training start the recon-learned part has zero contribution and
         V3 behaves identically to V1. As g grows during training, the
         recon term gradually opens up. Mirrors the Flamingo gate trick.

    Initialization (chicken-and-egg fix; PLAN §16.5 D-19):

        W_hebbian = 0                  (zero buffer)
        g         = 0                  (Flamingo gate closed at start)
        W_learned = 0.01 · N(0, 1)     (SMALL RANDOM, not zero)

    Why W_learned must be non-zero
    ------------------------------
    With g=0 and W_learned=0, both gradients are mathematically exact zero:

        ∂L/∂W_learned = ∂L/∂W · tanh(g)        = 0   (g=0)
        ∂L/∂g         = ∂L/∂W · sech²(g) · W_l = 0   (W_l=0)

    Neither AdamW's epsilon nor numerical noise can break this — the
    1st-moment estimate stays at zero forever. We observed this directly
    in Day 4c: 1 epoch of MNIST kept g exactly at 0.000000 while
    reconstruction loss diverged (0.05 → 15108) because W_hebbian grew
    arbitrarily without ever being corrected by W_learned.

    The fix: a tiny random W_learned. tanh(0) · W_learned is still 0 at
    init (so Flamingo's "no perturbation at start" property holds), but
    ∂L/∂g = ∂L/∂W · 1 · W_learned is now non-zero, so g receives gradient
    and starts growing → tanh(g) > 0 → W_learned receives gradient →
    symmetry broken.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        eta: float = 0.01,
        decay: float = 0.001,
        w_max: float = 1.0,
        learned_init_std: float = 0.01,
    ) -> None:
        super().__init__(hidden_dim, eta=eta, decay=decay, w_max=w_max)
        # Small random init for W_learned to break the chicken-and-egg
        # symmetry (see class docstring, PLAN §16.5 D-19).
        self.W_learned = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim) * learned_init_std
        )
        self.g = nn.Parameter(torch.zeros(()))  # scalar, learnable

    @property
    def W(self) -> Tensor:
        # Effective W = Hebbian buffer + gated learned matrix.
        return self.W_hebbian + torch.tanh(self.g) * self.W_learned

    # hebbian_update inherited from ACCv1Hebbian — updates W_hebbian only.


class CrossAttnAdapter(nn.Module):
    """B4 baseline: joint-trained bidirectional linear bridge.

    Why "CrossAttn" name?
    ---------------------
    PLAN §6 originally specified 1-head bidirectional cross-attention. In our
    setup hidden_L and hidden_R are single (B, D) vectors (no token sequence),
    so a 1-head attention degenerates: the softmax over a single element is
    always 1, and cross-attention reduces to a linear bridge. We keep the
    `CrossAttnAdapter` name for consistency with PLAN §17.3, but the actual
    implementation is a bidirectional linear projection with residual:

        h_L' = h_L + h_R @ W
        h_R' = h_R + h_L @ W.T

    KEY DESIGN: identical W shape and parameter count as ACCv2Recon
    (one (D, D) matrix, 4096 params for D=64). The ONLY difference vs the
    ACC variants is the *learning signal*:

        - ACC's W is updated by reconstruction loss / Hebbian rule,
          decoupled from the classification head (γ policy).
        - B4's W is updated only by backprop on the classification loss,
          jointly with the CNN backbones and classifier.

    This isolates the central hypothesis: does ACC's *separated* learning
    signal (recon loss + Hebbian) recover better hidden_R from hidden_L
    than B4's *joint* learning signal (classification loss only)?

    Usage in train.py:
        h_L = left_cnn(x_L); h_R = right_cnn(x_R)
        h_L_prime, h_R_prime = adapter(h_L, h_R)
        logits = classifier(h_L_prime, h_R_prime)   # passes through adapter
        loss = F.cross_entropy(logits, y)
        loss.backward()    # adapter.W gets grads from classification only

    Initialization: W = 0 → adapter is identity at step 0 → CNN/classifier
    train normally before W has any effect.
    """

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))

    def forward(
        self, hidden_L: Tensor, hidden_R: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Bidirectional cross-influence with residual.

        Args:
            hidden_L: (B, D)
            hidden_R: (B, D)
        Returns:
            (h_L_prime, h_R_prime), each (B, D).
        """
        # h_R contributes to h_L via W; h_L contributes to h_R via W.T.
        # Same W matrix in both directions → mirrors ACC's single-W convention.
        h_L_prime = hidden_L + hidden_R @ self.W
        h_R_prime = hidden_R + hidden_L @ self.W.T
        return h_L_prime, h_R_prime
