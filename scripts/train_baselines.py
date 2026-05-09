#!/usr/bin/env python
"""Train all variants for N epochs and report a results table.

Trains 7 variants (B1, B2a, B3, B4, V1, V2, V3) on full MNIST with the
same seed and identical dataloaders. After training, B2a's model is also
evaluated in left-only mode to extract B2(b) without separate training
(PLAN §6.1, option B — "shared training, two eval modes").

Usage (from project root, with venv activated):

    python scripts/train_baselines.py
    python scripts/train_baselines.py --epochs 1 --seed 42
    python scripts/train_baselines.py --epochs 3 --seed 42 --variants B3,B4,V3

Notes:
- 1 epoch on RTX 3070 Ti is ~30-60s per variant. 7 variants ≈ 5-7 minutes.
- For the full 5-seed sweep, see Day 7. This script is the Day 5 smoke check.
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

from split_mnist.data import make_loaders
from split_mnist.train import (
    TrainConfig,
    evaluate_left_only,
    train_one_run,
)


DEFAULT_VARIANTS = ("B1", "B2a", "B3", "B4", "V1", "V2", "V3")
PRINT_ORDER = ("B1", "B2a", "B2b", "B3", "B4", "V1", "V2", "V3")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(DEFAULT_VARIANTS),
        help=f"Comma-separated list. Default: {','.join(DEFAULT_VARIANTS)}",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda | cpu. Default: auto-detect.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== train_baselines: {len(variants)} variants × {args.epochs} epochs"
          f" × seed={args.seed} on {device_str} ===")
    print(f"variants: {variants}\n")

    # Build loaders ONCE so all variants see identical data splits.
    train_loader, val_loader, test_loader = make_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # variant -> dict(val, test, time, ...)
    results: dict[str, dict] = {}

    for variant in variants:
        print(f"\n{'-' * 70}")
        print(f"Training {variant}")
        print(f"{'-' * 70}")

        cfg = TrainConfig(
            variant=variant,
            seed=args.seed,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            log_every=200,  # quieter logs
            device=device_str,
        )

        t0 = time.time()
        result = train_one_run(
            cfg,
            _data_loaders=(train_loader, val_loader, test_loader),
        )
        elapsed = time.time() - t0

        results[variant] = {
            "val": result["best_val_acc"],
            "test": result["best_test_acc"],
            "time": elapsed,
            "metrics": result["final_metrics"],
            "note": "",
        }

        # B2a: derive B2b at no extra training cost (PLAN §6.1).
        if variant == "B2a":
            b2b_test = evaluate_left_only(
                result["model"],
                test_loader,
                device=device_str,
            )
            results["B2b"] = {
                "val": float("nan"),  # no separate val pass; eval-only
                "test": b2b_test,
                "time": 0.0,
                "metrics": {},
                "note": "(shared B2a model, eval=left_only)",
            }

    # ---------------- Results table ----------------
    print(f"\n\n{'=' * 76}")
    print(f"=== RESULTS  (epochs={args.epochs}, seed={args.seed}, "
          f"device={device_str}) ===")
    print(f"{'=' * 76}")
    header = f"{'variant':<10} {'val_acc':>10} {'test_acc':>10} {'time(s)':>10}  note"
    print(header)
    print("-" * 76)

    for v in PRINT_ORDER:
        if v not in results:
            continue
        r = results[v]
        val = r["val"]
        val_str = f"{val:.4f}" if val == val else "  --  "  # NaN-check
        note = r["note"]
        # For V3, also show g if available (D-19 monitoring).
        if v == "V3" and "g" in r["metrics"]:
            note = f"g={r['metrics']['g']:+.4f}"
        print(
            f"{v:<10} {val_str:>10} {r['test']:>10.4f}"
            f" {r['time']:>10.1f}  {note}"
        )

    print(f"{'=' * 76}\n")

    # ---------------- Quick sanity comparisons ----------------
    if "V3" in results and "B4" in results:
        print(f"V3 vs B4 (cls path):  test_acc {results['V3']['test']:.4f}"
              f" vs {results['B4']['test']:.4f}  "
              f"(Δ={results['V3']['test'] - results['B4']['test']:+.4f})")
    if "B1" in results and "B3" in results:
        print(f"B1 vs B3 (split cost): test_acc {results['B1']['test']:.4f}"
              f" vs {results['B3']['test']:.4f}  "
              f"(Δ={results['B1']['test'] - results['B3']['test']:+.4f})")
    if "B2a" in results and "B2b" in results:
        print(f"B2a vs B2b:           test_acc {results['B2a']['test']:.4f}"
              f" vs {results['B2b']['test']:.4f}  "
              f"(Δ={results['B2a']['test'] - results['B2b']['test']:+.4f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
