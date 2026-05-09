"""Loss functions for SPLIT-MNIST training.

Reconstruction loss and Hebbian updates live on ACC classes themselves
(see acc.py); this file only houses the classification loss for now.

See PLAN §17.4.
"""
from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def classification_loss(logits: Tensor, y: Tensor) -> Tensor:
    """Standard cross-entropy. Wrapped for naming consistency only."""
    return F.cross_entropy(logits, y)
