"""Faithfulness measurements for SPLIT-MNIST.

Implements PLAN §5 measurements:

  #1 measure_task_accuracy        — wrapper around train._evaluate
  #2 measure_cross_activation     ★ core hypothesis test
  #3 measure_causal_coupling
  #4 measure_position_invariance  (called separately, post-training)

Variant applicability matrix (see PLAN §6):

  variant   | #1   | #2                                     | #3   | #4
  ----------|------|----------------------------------------|------|-----
  B1        | OK   | N/A (h_L/h_R not separated)            | N/A  | N/A
  B2a/B3    | OK   | partial: acc_real, acc_ablated only    | OK   | N/A
  B4        | OK   | OK (cosine semantics differ from ACC)  | OK   | N/A
  V1/V2/V3  | OK   | OK                                     | OK   | OK
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader


# Variants for which #2 (cross-activation reconstruction) yields a meaningful hat.
ACC_LIKE_VARIANTS = ("V1", "V2", "V3", "B4")

# Variants for which #2 partial (acc_real, acc_ablated) is computable.
SPLIT_VARIANTS = ("B2a", "B3", "B4", "V1", "V2", "V3")


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------


@torch.no_grad()
def _compute_global_mean_hidden(
    model: dict,
    loader: DataLoader,
    device: torch.device,
    side: str = "R",
) -> Tensor:
    """Mean of `side` hidden across the entire loader. Returns shape (D,)."""
    cnn_key = "right" if side == "R" else "left"
    cnn = model[cnn_key]
    cnn.eval()
    sum_h = None
    n = 0
    for x_L, x_R, _ in loader:
        x = (x_R if side == "R" else x_L).to(device, non_blocking=True)
        h = cnn(x)
        sum_h = h.sum(0) if sum_h is None else sum_h + h.sum(0)
        n += h.shape[0]
    if sum_h is None:
        raise ValueError("loader yielded no batches")
    return sum_h / n


def _classify(model: dict, variant: str, h_L: Tensor, h_R: Tensor) -> Tensor:
    """Apply the variant's classification head to (h_L, h_R) → logits.

    Crucial: this function is *variant-aware* but does NOT re-run the CNNs.
    It accepts already-extracted hiddens, possibly substituted (ablated /
    reconstructed). Used for measurement #2.

    For B4, the adapter is part of the main forward path; we apply it here
    so that the classifier sees the post-adapter hiddens (mirroring what
    happens during training).
    """
    if variant == "B1":
        raise NotImplementedError("B1 has no separate h_L/h_R; use full forward.")
    if variant == "B2a":
        return model["classifier"].avg_logits(h_L, h_R)
    if variant == "B4":
        h_L_p, h_R_p = model["adapter"](h_L, h_R)
        return model["classifier"](h_L_p, h_R_p)
    # B3, V1, V2, V3 — γ policy: classifier sees raw concat.
    return model["classifier"](h_L, h_R)


def _reconstruct_h_R(
    model: dict, variant: str, h_L: Tensor, mu_R: Tensor
) -> Tensor:
    """Reconstruct ĥ_R from h_L (and the right-side mean for B4)."""
    if variant in ("V1", "V2", "V3"):
        # ACC's bidirectional matrix: ĥ_R = h_L @ W.T  (i.e. W · h_L for col vec).
        return model["acc"].forward_LR(h_L)
    if variant == "B4":
        # Pass mean-ablated h_R into the adapter; take its h_R' output.
        mu_R_batch = mu_R.unsqueeze(0).expand_as(h_L)
        _, h_R_prime = model["adapter"](h_L, mu_R_batch)
        return h_R_prime
    raise ValueError(
        f"Cross-activation reconstruction not defined for variant={variant!r}"
    )


def _reconstruct_h_L(
    model: dict, variant: str, h_R: Tensor, mu_L: Tensor
) -> Tensor:
    """Reconstruct ĥ_L from h_R."""
    if variant in ("V1", "V2", "V3"):
        return model["acc"].forward_RL(h_R)
    if variant == "B4":
        mu_L_batch = mu_L.unsqueeze(0).expand_as(h_R)
        h_L_prime, _ = model["adapter"](mu_L_batch, h_R)
        return h_L_prime
    raise ValueError(
        f"Cross-activation reconstruction not defined for variant={variant!r}"
    )


# -----------------------------------------------------------------------------
# Measurement #1 — task accuracy
# -----------------------------------------------------------------------------


@torch.no_grad()
def measure_task_accuracy(
    model: dict,
    loader: DataLoader,
    variant: str,
    device: torch.device | str = "cpu",
) -> float:
    """Standard classification accuracy (PLAN §5.1).

    Mirrors train._evaluate but exposed as public API. Used for repeated
    eval calls outside training (e.g. after loading a checkpoint).
    """
    device_t = torch.device(device)
    for m in model.values():
        m.eval()

    if variant == "B1":
        correct = 0
        total = 0
        for x_L, x_R, y in loader:
            x_L = x_L.to(device_t)
            x_R = x_R.to(device_t)
            y = y.to(device_t)
            x_full = torch.cat([x_L, x_R], dim=-1)
            preds = model["single_cnn"](x_full).argmax(-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        return correct / max(total, 1)

    correct = 0
    total = 0
    for x_L, x_R, y in loader:
        x_L = x_L.to(device_t)
        x_R = x_R.to(device_t)
        y = y.to(device_t)
        h_L = model["left"](x_L)
        h_R = model["right"](x_R)
        logits = _classify(model, variant, h_L, h_R)
        preds = logits.argmax(-1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


# -----------------------------------------------------------------------------
# Measurement #2 — cross-activation faithfulness  ★ CORE
# -----------------------------------------------------------------------------


@torch.no_grad()
def measure_cross_activation(
    model: dict,
    loader: DataLoader,
    variant: str,
    device: torch.device | str = "cpu",
    direction: str = "right_ablation",
) -> dict:
    """Cross-activation faithfulness (PLAN §5.2). ★ Core hypothesis test.

    Procedure (`direction="right_ablation"`):

      1. Compute global mean μ_R over the loader.
      2. For each batch (x_L, x_R, y):
         a. h_L = left(x_L), h_R = right(x_R)
         b. ĥ_R = ACC.forward_LR(h_L)   (or B4 adapter output)
         c. cosine = cos(ĥ_R, h_R)      per-sample, then averaged
         d. Three accuracies:
              - acc_real          := classifier(h_L, h_R 진짜)
              - acc_ablated       := classifier(h_L, μ_R)
              - acc_reconstructed := classifier(h_L, ĥ_R)
      3. Aggregate.

    Returns keys (always present, NaN where N/A):

      "cosine":          mean cosine similarity over samples
      "cosine_std":      std cosine similarity
      "acc_real":        classifier accuracy with real h_R
      "acc_ablated":     classifier accuracy with μ_R substituted
      "acc_reconstructed": classifier accuracy with reconstructed ĥ_R
      "n_samples":       total samples seen
      "applicable_recon": True if variant ∈ ACC_LIKE_VARIANTS
      "applicable_split": True if variant ∈ SPLIT_VARIANTS
      "direction":       echo of input
    """
    if direction not in ("right_ablation", "left_ablation"):
        raise ValueError(f"direction must be 'right_ablation' or 'left_ablation'")

    device_t = torch.device(device)
    nan = float("nan")

    if variant not in SPLIT_VARIANTS:
        return {
            "cosine": nan,
            "cosine_std": nan,
            "acc_real": nan,
            "acc_ablated": nan,
            "acc_reconstructed": nan,
            "n_samples": 0,
            "applicable_recon": False,
            "applicable_split": False,
            "direction": direction,
        }

    for m in model.values():
        m.eval()

    # Precompute the side-mean we'll ablate to.
    side = "R" if direction == "right_ablation" else "L"
    mu = _compute_global_mean_hidden(model, loader, device_t, side=side)

    cosines: list[Tensor] = []
    correct_real = 0
    correct_ablated = 0
    correct_reconstructed = 0
    total = 0
    has_recon = variant in ACC_LIKE_VARIANTS

    for x_L, x_R, y in loader:
        x_L = x_L.to(device_t)
        x_R = x_R.to(device_t)
        y = y.to(device_t)
        h_L = model["left"](x_L)
        h_R = model["right"](x_R)

        if direction == "right_ablation":
            mu_R_b = mu.unsqueeze(0).expand_as(h_R)
            logits_real = _classify(model, variant, h_L, h_R)
            logits_ablated = _classify(model, variant, h_L, mu_R_b)
            if has_recon:
                hat_R = _reconstruct_h_R(model, variant, h_L, mu)
                cos = F.cosine_similarity(hat_R, h_R, dim=-1)
                cosines.append(cos)
                logits_recon = _classify(model, variant, h_L, hat_R)
        else:  # left_ablation
            mu_L_b = mu.unsqueeze(0).expand_as(h_L)
            logits_real = _classify(model, variant, h_L, h_R)
            logits_ablated = _classify(model, variant, mu_L_b, h_R)
            if has_recon:
                hat_L = _reconstruct_h_L(model, variant, h_R, mu)
                cos = F.cosine_similarity(hat_L, h_L, dim=-1)
                cosines.append(cos)
                logits_recon = _classify(model, variant, hat_L, h_R)

        correct_real += (logits_real.argmax(-1) == y).sum().item()
        correct_ablated += (logits_ablated.argmax(-1) == y).sum().item()
        if has_recon:
            correct_reconstructed += (logits_recon.argmax(-1) == y).sum().item()
        total += y.size(0)

    if has_recon:
        cos_all = torch.cat(cosines)
        cos_mean = cos_all.mean().item()
        cos_std = cos_all.std().item()
        acc_recon = correct_reconstructed / max(total, 1)
    else:
        cos_mean = nan
        cos_std = nan
        acc_recon = nan

    return {
        "cosine": cos_mean,
        "cosine_std": cos_std,
        "acc_real": correct_real / max(total, 1),
        "acc_ablated": correct_ablated / max(total, 1),
        "acc_reconstructed": acc_recon,
        "n_samples": total,
        "applicable_recon": has_recon,
        "applicable_split": True,
        "direction": direction,
    }


# -----------------------------------------------------------------------------
# Measurement #3 — causal coupling
# -----------------------------------------------------------------------------


@torch.no_grad()
def measure_causal_coupling(
    model: dict,
    loader: DataLoader,
    variant: str,
    device: torch.device | str = "cpu",
    epsilons: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0),
    seed: int = 0,
) -> dict:
    """Causal coupling: noise added to h_L should propagate to classifier
    accuracy (PLAN §5.3).

    Returns:
        "epsilons":   the input epsilons
        "accuracies": list[float], same length as epsilons
        "ias_at_0p5": (acc[ε=0] - acc[ε=0.5]) / acc[ε=0]   (relative drop)
        "applicable": True for split variants (B2a, B3, B4, V1, V2, V3)
        "n_samples":  per-epsilon total
    """
    if variant == "B1" or variant not in SPLIT_VARIANTS:
        return {
            "epsilons": list(epsilons),
            "accuracies": [float("nan")] * len(epsilons),
            "ias_at_0p5": float("nan"),
            "applicable": False,
            "n_samples": 0,
        }

    device_t = torch.device(device)
    for m in model.values():
        m.eval()

    g = torch.Generator(device=device_t).manual_seed(seed)

    accs: list[float] = []
    last_total = 0
    for eps in epsilons:
        correct = 0
        total = 0
        for x_L, x_R, y in loader:
            x_L = x_L.to(device_t)
            x_R = x_R.to(device_t)
            y = y.to(device_t)
            h_L = model["left"](x_L)
            h_R = model["right"](x_R)
            if eps > 0:
                noise = torch.randn(h_L.shape, generator=g, device=device_t) * eps
                h_L = h_L + noise
            logits = _classify(model, variant, h_L, h_R)
            preds = logits.argmax(-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        accs.append(correct / max(total, 1))
        last_total = total

    # IAS at ε=0.5
    if 0.5 in epsilons and 0.0 in epsilons:
        i0 = list(epsilons).index(0.0)
        i5 = list(epsilons).index(0.5)
        ias = (accs[i0] - accs[i5]) / max(accs[i0], 1e-9)
    else:
        ias = float("nan")

    return {
        "epsilons": list(epsilons),
        "accuracies": accs,
        "ias_at_0p5": ias,
        "applicable": True,
        "n_samples": last_total,
    }


# -----------------------------------------------------------------------------
# Measurement #4 — position invariance (post 5-seed training)
# -----------------------------------------------------------------------------


def measure_position_invariance(W_list: list[Tensor]) -> dict:
    """Procrustes alignment of W matrices across seeds (PLAN §5.4).

    Args:
        W_list: list of (D, D) tensors, one per seed. Length ≥ 2.

    Returns:
        "procrustes_corr_mean":     mean Pearson r across pairs (W_0, W_i)
        "procrustes_corr_per_seed": list of r values, length len(W_list) - 1
        "n_seeds":                  len(W_list)

    Interpretation (PLAN §5.5):
        r ≤ 0.3 → 위치 무관 OK (각 seed가 *다른 짝꿍 패턴*을 발견)
        r > 0.7 → 수렴 발견 (W가 seed와 무관한 *고정 구조*에 수렴 → 가설 약화)
        0.3~0.7 → 모호 구간
    """
    import numpy as np
    from scipy.linalg import orthogonal_procrustes

    if len(W_list) < 2:
        raise ValueError(f"Need ≥ 2 W matrices, got {len(W_list)}")

    ref = W_list[0].detach().cpu().numpy()
    correlations: list[float] = []
    for W in W_list[1:]:
        W_np = W.detach().cpu().numpy()
        # Find orthogonal R minimizing ||W_np · R - ref||_F.
        R, _ = orthogonal_procrustes(W_np, ref)
        W_aligned = W_np @ R
        # Pearson correlation on flattened entries.
        c = float(np.corrcoef(W_aligned.flatten(), ref.flatten())[0, 1])
        correlations.append(c)

    return {
        "procrustes_corr_mean": float(sum(correlations) / len(correlations)),
        "procrustes_corr_per_seed": correlations,
        "n_seeds": len(W_list),
    }


def measure_random_baseline_invariance(
    hidden_dim: int = 64,
    n_seeds: int = 5,
    base_seed: int = 100,
) -> dict:
    """Establish the random-init Procrustes baseline for D-21 calibration.

    Generates n_seeds × isotropic random (D, D) matrices and measures
    `measure_position_invariance` over them. The resulting r is the
    *noise floor* — any trained W's r must be distinguishably *below*
    this baseline (Δ ≥ 0.15) to claim "위치 무관 OK".

    See PLAN §16.5 D-21 for the full rationale.

    Args:
        hidden_dim: ACC W dimension (default 64).
        n_seeds:    number of random matrices (default 5, matching the
                    per-seed count used for the trained measurement).
        base_seed:  starting seed; matrices use base_seed, base_seed+1, ...
                    Default 100 to avoid collision with training seeds 42-46.

    Returns:
        Same shape as `measure_position_invariance`'s output.
    """
    W_list = []
    for i in range(n_seeds):
        g = torch.Generator().manual_seed(base_seed + i)
        # Plain N(0, 1) — same isotropy as Day 6's empirical observation.
        W = torch.randn(hidden_dim, hidden_dim, generator=g)
        W_list.append(W)
    return measure_position_invariance(W_list)
