"""Bilateral CNN architectures for SPLIT-MNIST.

- HalfCNN: one half (used twice for left/right with independent weights).
- Classifier: takes concat[hidden_L, hidden_R] and predicts class.
- IndependentClassifiers: B2(a/b) baseline — two parallel classifiers,
  one per hemisphere, used independently.
- SingleCNN: B1 baseline (sees the full 28x28 image).

All shapes follow PLAN §3.2 and §6.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HalfCNN(nn.Module):
    """One hemisphere. Input is half of the MNIST image.

    Input:  (B, 1, 28, 14)
    Output: (B, hidden_dim)   [default hidden_dim=64]

    Architecture (PLAN §3.2 (2)):
        Conv(1->16, 3x3, pad=1) -> ReLU -> MaxPool 2x2
        Conv(16->32, 3x3, pad=1) -> ReLU -> MaxPool 2x2
        Flatten -> Linear(32*7*3 -> hidden_dim) -> ReLU
    """

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # After two 2x2 max pools on (28, 14):
        #   (28, 14) -> (14, 7) -> (7, 3)   [floor div]
        # Flattened: 32 * 7 * 3 = 672
        self.fc = nn.Linear(32 * 7 * 3, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, 1, 28, 14)
        x = F.relu(self.conv1(x))         # (B, 16, 28, 14)
        x = F.max_pool2d(x, 2)            # (B, 16, 14, 7)
        x = F.relu(self.conv2(x))         # (B, 32, 14, 7)
        x = F.max_pool2d(x, 2)            # (B, 32, 7, 3)
        x = x.flatten(start_dim=1)        # (B, 672)
        x = F.relu(self.fc(x))            # (B, hidden_dim)
        return x


class Classifier(nn.Module):
    """Combine left/right hidden vectors and classify.

    Input:  hidden_L (B, hidden_dim), hidden_R (B, hidden_dim)
    Output: logits (B, n_classes)

    PLAN §3.2 (4): the classifier sees raw concat during training (alpha mode).
    Cross-activation evaluation will replace one half with the ACC reconstruction
    at evaluation time only (beta mode), without re-training.
    """

    def __init__(self, hidden_dim: int = 64, n_classes: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, hidden_L: Tensor, hidden_R: Tensor) -> Tensor:
        x = torch.cat([hidden_L, hidden_R], dim=-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class IndependentClassifiers(nn.Module):
    """B2 baseline: two parallel classifiers, one per hemisphere.

    Used by both B2(a) and B2(b) baselines (PLAN §6.1):
      - B2(a) Independent-avg: (logits_L + logits_R) / 2, then argmax.
      - B2(b) Left-only: logits_L only, then argmax.

    Both share the same trained weights — see PLAN §6.1 footnote on
    "B2(b) is derived from B2(a) by changing only the eval mode."

    Architecture: same as Classifier's two-stage MLP, but applied
    independently to each hemisphere's hidden vector.

    Forward returns the two logit tensors as a tuple. Use
    `IndependentClassifiers.avg_logits()` for B2(a) eval, or take the
    first element for B2(b) eval.
    """

    def __init__(self, hidden_dim: int = 64, n_classes: int = 10) -> None:
        super().__init__()
        self.left = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )
        self.right = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(
        self, hidden_L: Tensor, hidden_R: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return (logits_L, logits_R), each (B, n_classes)."""
        return self.left(hidden_L), self.right(hidden_R)

    def avg_logits(self, hidden_L: Tensor, hidden_R: Tensor) -> Tensor:
        """Convenience: return the per-class logit average for B2(a) eval."""
        logits_L, logits_R = self.forward(hidden_L, hidden_R)
        return (logits_L + logits_R) / 2


class SingleCNN(nn.Module):
    """B1 baseline: a single CNN that sees the full MNIST image.

    Input:  (B, 1, 28, 28)
    Output: logits (B, n_classes)

    Reference *upper bound* — what's achievable when nothing is split.
    See PLAN §6 (B1 row).
    """

    def __init__(self, hidden_dim: int = 64, n_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # After two 2x2 max pools on (28, 28):
        #   (28, 28) -> (14, 14) -> (7, 7)
        # Flattened: 32 * 7 * 7 = 1568
        self.fc1 = nn.Linear(32 * 7 * 7, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, 1, 28, 28)
        x = F.relu(self.conv1(x))         # (B, 16, 28, 28)
        x = F.max_pool2d(x, 2)            # (B, 16, 14, 14)
        x = F.relu(self.conv2(x))         # (B, 32, 14, 14)
        x = F.max_pool2d(x, 2)            # (B, 32, 7, 7)
        x = x.flatten(start_dim=1)        # (B, 1568)
        x = F.relu(self.fc1(x))           # (B, hidden_dim)
        return self.fc2(x)                # (B, n_classes)
