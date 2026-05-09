"""Joint training loop for all SPLIT-MNIST variants.

Single entry point `train_one_run(cfg)` handles 6 variants (B2 deferred to Day 5):

    B1  — SingleCNN on the full 28x28 image            (upper-bound reference)
    B3  — Two HalfCNN + Classifier (raw concat)        (multimodal baseline)
    B4  — + CrossAttnAdapter trained jointly           (post-hoc adapter)
    V1  — + ACCv1Hebbian (Hebbian only, no backprop)
    V2  — + ACCv2Recon   (reconstruction loss only)
    V3  — + ACCv3Combined (Hebbian + reconstruction)   ★ hypothesized form

Variant-specific forward/backward flows are documented in
`_step` below.

See PLAN §3, §4, §6, §9, §17.5.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .acc import (
    ACCv1Hebbian,
    ACCv2Recon,
    ACCv3Combined,
    CrossAttnAdapter,
)
from .data import make_loaders
from .losses import classification_loss
from .networks import Classifier, HalfCNN, SingleCNN


VALID_VARIANTS = ("B1", "B3", "B4", "V1", "V2", "V3")
ACC_VARIANTS = ("V1", "V2", "V3")
RECON_VARIANTS = ("V2", "V3")
HEBBIAN_VARIANTS = ("V1", "V3")


@dataclass
class TrainConfig:
    """All hyperparameters in one place. See PLAN §9."""

    # Variant — picks the model + learning signal
    variant: str = "V3"

    # Optimizer
    seed: int = 42
    n_epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # ACC reconstruction loss weight (V2, V3 only)
    beta_recon: float = 1.0

    # Hebbian (V1, V3 only)
    hebbian_eta: float = 0.01
    hebbian_decay: float = 0.001
    hebbian_w_max: float = 1.0

    # Model dims
    hidden_dim: int = 64
    n_classes: int = 10

    # Early stopping
    early_stop_patience: int = 3

    # Data
    data_root: str = "./data"
    val_size: int = 10_000
    num_workers: int = 0

    # IO
    out_dir: str = "runs/"
    log_every: int = 100  # log loss every N steps

    # Device override (None = auto-detect)
    device: str | None = None

    def __post_init__(self) -> None:
        if self.variant not in VALID_VARIANTS:
            raise ValueError(
                f"variant must be one of {VALID_VARIANTS}, got {self.variant!r}"
            )


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set seeds for python, numpy, and torch (CPU+CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_of(cfg: TrainConfig) -> torch.device:
    if cfg.device is not None:
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Model construction
# -----------------------------------------------------------------------------


def build_model(cfg: TrainConfig) -> dict:
    """Construct the model components for a given variant.

    Returns a dict whose keys depend on variant:

        B1       -> {"single_cnn": SingleCNN}
        B3       -> {"left", "right", "classifier"}
        B4       -> {"left", "right", "classifier", "adapter": CrossAttnAdapter}
        V1/V2/V3 -> {"left", "right", "classifier", "acc": ACC*}
    """
    if cfg.variant == "B1":
        return {
            "single_cnn": SingleCNN(
                hidden_dim=cfg.hidden_dim, n_classes=cfg.n_classes
            )
        }

    # All non-B1 variants share the bilateral backbone + classifier.
    components: dict = {
        "left": HalfCNN(hidden_dim=cfg.hidden_dim),
        "right": HalfCNN(hidden_dim=cfg.hidden_dim),
        "classifier": Classifier(
            hidden_dim=cfg.hidden_dim, n_classes=cfg.n_classes
        ),
    }

    if cfg.variant == "B3":
        return components

    if cfg.variant == "B4":
        components["adapter"] = CrossAttnAdapter(hidden_dim=cfg.hidden_dim)
        return components

    # ACC variants (V1/V2/V3)
    if cfg.variant == "V1":
        components["acc"] = ACCv1Hebbian(
            hidden_dim=cfg.hidden_dim,
            eta=cfg.hebbian_eta,
            decay=cfg.hebbian_decay,
            w_max=cfg.hebbian_w_max,
        )
    elif cfg.variant == "V2":
        components["acc"] = ACCv2Recon(hidden_dim=cfg.hidden_dim)
    elif cfg.variant == "V3":
        components["acc"] = ACCv3Combined(
            hidden_dim=cfg.hidden_dim,
            eta=cfg.hebbian_eta,
            decay=cfg.hebbian_decay,
            w_max=cfg.hebbian_w_max,
        )

    return components


def _all_learnable_params(model: dict) -> list:
    """Flatten all nn.Parameters across model components for the optimizer."""
    params = []
    for m in model.values():
        params.extend(p for p in m.parameters() if p.requires_grad)
    return params


def _move_to(model: dict, device: torch.device) -> dict:
    for k in model:
        model[k] = model[k].to(device)
    return model


def _set_train(model: dict) -> None:
    for m in model.values():
        m.train()


def _set_eval(model: dict) -> None:
    for m in model.values():
        m.eval()


# -----------------------------------------------------------------------------
# Forward — variant-specific
# -----------------------------------------------------------------------------


def _forward_logits(
    model: dict,
    x_L: Tensor,
    x_R: Tensor,
    cfg: TrainConfig,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """Compute logits for the given variant.

    Returns:
        (logits, h_L, h_R)
        - logits: (B, n_classes)
        - h_L, h_R: (B, hidden_dim) for non-B1 variants; None for B1.
          (Used downstream by recon loss / Hebbian update.)
    """
    if cfg.variant == "B1":
        x_full = torch.cat([x_L, x_R], dim=-1)  # (B, 1, 28, 28)
        return model["single_cnn"](x_full), None, None

    h_L = model["left"](x_L)   # (B, D)
    h_R = model["right"](x_R)  # (B, D)

    if cfg.variant == "B4":
        # Adapter is in the main path; classifier sees the modified hiddens.
        h_L_prime, h_R_prime = model["adapter"](h_L, h_R)
        logits = model["classifier"](h_L_prime, h_R_prime)
        return logits, h_L, h_R  # h_L/h_R for downstream — unused by B4

    # B3, V1, V2, V3 — classifier sees raw concat (γ policy: ACC is not in main path).
    logits = model["classifier"](h_L, h_R)
    return logits, h_L, h_R


# -----------------------------------------------------------------------------
# One training step
# -----------------------------------------------------------------------------


def _step(
    model: dict,
    optimizer: torch.optim.Optimizer,
    x_L: Tensor,
    x_R: Tensor,
    y: Tensor,
    cfg: TrainConfig,
) -> dict[str, float]:
    """Run one training step. Returns scalar metrics for logging.

    The variant-specific behavior:
      - B1, B3      : classification loss only.
      - B4          : classification loss; adapter.W gets grads through it.
      - V1          : classification loss; Hebbian update on detached hiddens.
      - V2          : classification + β·reconstruction (ACC, detached hiddens).
      - V3          : classification + β·reconstruction + Hebbian.
    """
    logits, h_L, h_R = _forward_logits(model, x_L, x_R, cfg)
    loss_cls = classification_loss(logits, y)

    metrics: dict[str, float] = {"loss_cls": loss_cls.item()}

    # Add reconstruction loss if applicable. ACC sees DETACHED hiddens so the
    # backbones are not perturbed by the recon signal (PLAN §4.5 (5)).
    if cfg.variant in RECON_VARIANTS:
        assert h_L is not None and h_R is not None
        loss_recon = model["acc"].reconstruction_loss(h_L.detach(), h_R.detach())
        loss_total = loss_cls + cfg.beta_recon * loss_recon
        metrics["loss_recon"] = loss_recon.item()
    else:
        loss_total = loss_cls

    metrics["loss"] = loss_total.item()

    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()

    # Hebbian update — outside the autograd graph.
    if cfg.variant in HEBBIAN_VARIANTS:
        assert h_L is not None and h_R is not None
        # `hebbian_update` is decorated with @torch.no_grad in the ACC class;
        # we still pass detached tensors for clarity.
        model["acc"].hebbian_update(h_L.detach(), h_R.detach())

    # For diagnostic purposes (used by Day 4c smoke check on V3):
    if cfg.variant == "V3":
        metrics["g"] = model["acc"].g.item()

    return metrics


# -----------------------------------------------------------------------------
# Evaluation (Task accuracy = Measurement #1, PLAN §5.1)
# -----------------------------------------------------------------------------


@torch.no_grad()
def _evaluate(
    model: dict,
    loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
) -> float:
    """Compute classification accuracy on `loader`."""
    _set_eval(model)
    correct = 0
    total = 0
    for x_L, x_R, y in loader:
        x_L = x_L.to(device, non_blocking=True)
        x_R = x_R.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits, _, _ = _forward_logits(model, x_L, x_R, cfg)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------


def train_one_run(cfg: TrainConfig, _data_loaders: tuple | None = None) -> dict:
    """Run a single training experiment.

    Args:
        cfg: TrainConfig
        _data_loaders: optional (train_loader, val_loader, test_loader) for
            tests / smoke runs. If None, loaders are built from cfg.

    Returns:
        dict with keys:
            "val_acc_curve":  list of val accuracies, one per epoch
            "best_val_acc":   best val accuracy seen
            "best_test_acc":  test accuracy at the best-val epoch
            "final_metrics":  metrics from the last training step
    """
    set_seed(cfg.seed)
    device = _device_of(cfg)

    # Data
    if _data_loaders is None:
        train_loader, val_loader, test_loader = make_loaders(
            root=cfg.data_root,
            batch_size=cfg.batch_size,
            val_size=cfg.val_size,
            seed=cfg.seed,
            num_workers=cfg.num_workers,
        )
    else:
        train_loader, val_loader, test_loader = _data_loaders

    # Model
    model = build_model(cfg)
    model = _move_to(model, device)

    optimizer = torch.optim.AdamW(
        _all_learnable_params(model),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # Output dir
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_acc_curve: list[float] = []
    best_val_acc = -1.0
    best_test_acc = -1.0
    patience = 0
    final_metrics: dict[str, float] = {}

    for epoch in range(cfg.n_epochs):
        _set_train(model)
        last_step_metrics: dict[str, float] = {}

        for step_i, (x_L, x_R, y) in enumerate(train_loader):
            x_L = x_L.to(device, non_blocking=True)
            x_R = x_R.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            metrics = _step(model, optimizer, x_L, x_R, y, cfg)
            last_step_metrics = metrics

            if step_i % cfg.log_every == 0:
                msg = f"epoch={epoch} step={step_i} " + " ".join(
                    f"{k}={v:.4f}" for k, v in metrics.items()
                )
                print(msg)

        # End-of-epoch evaluation
        val_acc = _evaluate(model, val_loader, cfg, device)
        val_acc_curve.append(val_acc)
        final_metrics = last_step_metrics

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_test_acc = _evaluate(model, test_loader, cfg, device)
            patience = 0
        else:
            patience += 1

        print(
            f"epoch={epoch} val_acc={val_acc:.4f} "
            f"best_val={best_val_acc:.4f} best_test={best_test_acc:.4f} "
            f"patience={patience}/{cfg.early_stop_patience}"
        )

        if patience >= cfg.early_stop_patience:
            print(f"Early stop at epoch {epoch}.")
            break

    return {
        "val_acc_curve": val_acc_curve,
        "best_val_acc": best_val_acc,
        "best_test_acc": best_test_acc,
        "final_metrics": final_metrics,
    }
