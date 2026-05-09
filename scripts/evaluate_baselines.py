#!/usr/bin/env python
"""Train all variants for N epochs, then run measurements #2 and #3.

This is the **first end-to-end test of the central hypothesis**: does V3
recover hidden_R better than B4? Outputs three results tables:

    1. Task accuracy (= measurement #1)
    2. Cross-activation faithfulness (= measurement #2, both directions)
    3. Causal coupling — ε sweep (= measurement #3)

Plus the D-21 random Procrustes baseline (no training required) and
hypothesis-relevant comparisons (V3 vs B4, etc.).

Note: Measurement #4 (position invariance over trained W's) requires
multi-seed training and lives in Day 7's full sweep, not here.

Usage:
    python scripts/evaluate_baselines.py
    python scripts/evaluate_baselines.py --epochs 1 --seed 42
    python scripts/evaluate_baselines.py --variants V3,B4
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import torch

from split_mnist.data import make_loaders
from split_mnist.evaluate import (
    measure_causal_coupling,
    measure_cross_activation,
    measure_random_baseline_invariance,
)
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
        "--variants", type=str, default=",".join(DEFAULT_VARIANTS)
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def _fmt(x: float, w: int = 10, p: int = 4) -> str:
    """Right-align a float, or '  --' if NaN."""
    if x != x:  # NaN
        return f"{'  --':>{w}}"
    return f"{x:>{w}.{p}f}"


def main() -> int:
    args = _parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"=== evaluate_baselines: {len(variants)} variants × {args.epochs} epochs"
        f" × seed={args.seed} on {device_str} ===\n"
    )

    # ---- D-21 random baseline (before any training) ----
    print("--- D-21: Random-init Procrustes baseline (no training required) ---")
    rb = measure_random_baseline_invariance(hidden_dim=64, n_seeds=5)
    rb_mean = rb["procrustes_corr_mean"]
    print(f"r_random_baseline = {rb_mean:.4f}")
    rb_per = [round(r, 4) for r in rb["procrustes_corr_per_seed"]]
    print(f"  per-seed: {rb_per}")
    print(
        "  → Trained W's r must be ≤ this minus 0.15 to claim 위치 무관 OK.\n"
    )

    # ---- Build loaders ONCE ----
    train_loader, val_loader, test_loader = make_loaders(
        root=args.data_root, batch_size=args.batch_size, seed=args.seed
    )

    results: dict[str, dict] = {}

    for variant in variants:
        print(f"\n{'-' * 70}\nTraining {variant}\n{'-' * 70}")
        cfg = TrainConfig(
            variant=variant,
            seed=args.seed,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            log_every=200,
            device=device_str,
        )

        t0 = time.time()
        result = train_one_run(
            cfg,
            _data_loaders=(train_loader, val_loader, test_loader),
        )
        train_time = time.time() - t0

        # Measurements #2 (both directions) + #3.
        print(f"Measuring {variant}...")
        t0 = time.time()
        meas2_R = measure_cross_activation(
            result["model"], test_loader, variant,
            device=device_str, direction="right_ablation",
        )
        meas2_L = measure_cross_activation(
            result["model"], test_loader, variant,
            device=device_str, direction="left_ablation",
        )
        meas3 = measure_causal_coupling(
            result["model"], test_loader, variant,
            device=device_str,
            epsilons=(0.0, 0.1, 0.25, 0.5, 1.0, 2.0),
        )
        meas_time = time.time() - t0

        results[variant] = {
            "task_acc_test": result["best_test_acc"],
            "task_acc_val": result["best_val_acc"],
            "train_time": train_time,
            "meas_time": meas_time,
            "meas2_R": meas2_R,
            "meas2_L": meas2_L,
            "meas3": meas3,
            "model": result["model"],
            "final_metrics": result["final_metrics"],
            "note": "",
        }

        # B2b: derived from B2a model (PLAN §6.1, "shared model, two eval modes").
        if variant == "B2a":
            b2b_acc = evaluate_left_only(
                result["model"], test_loader, device=device_str
            )
            results["B2b"] = {
                "task_acc_test": b2b_acc,
                "task_acc_val": float("nan"),
                "train_time": 0.0,
                "meas_time": 0.0,
                "meas2_R": None,
                "meas2_L": None,
                "meas3": None,
                "final_metrics": {},
                "note": "(shared B2a model, eval=left_only)",
            }

    # ====================================================================
    # TABLE 1 — Task accuracy
    # ====================================================================
    print(f"\n\n{'=' * 80}")
    print("=== TABLE 1: Task accuracy ===")
    print(f"{'=' * 80}")
    print(
        f"{'variant':<10}{'val_acc':>12}{'test_acc':>12}{'train_t(s)':>12}  notes"
    )
    print("-" * 80)
    for v in PRINT_ORDER:
        if v not in results:
            continue
        r = results[v]
        val_str = _fmt(r["task_acc_val"], w=12)
        notes = r["note"]
        if v == "V3" and r.get("final_metrics"):
            g = r["final_metrics"].get("g")
            if g is not None:
                notes = f"g={g:+.4f}"
        print(
            f"{v:<10}{val_str}{r['task_acc_test']:>12.4f}"
            f"{r['train_time']:>12.1f}  {notes}"
        )

    # ====================================================================
    # TABLE 2 — Cross-activation faithfulness (right_ablation)
    # ====================================================================
    print(f"\n{'=' * 90}")
    print("=== TABLE 2: Cross-activation faithfulness — right_ablation ===")
    print("    (gate h_R, recover from h_L via ACC.forward_LR or B4 adapter)")
    print(f"{'=' * 90}")
    print(
        f"{'variant':<10}{'cosine':>10}{'acc_real':>10}{'acc_abl':>10}"
        f"{'acc_recon':>11}{'Δ(rec-abl)':>12}"
    )
    print("-" * 90)
    for v in PRINT_ORDER:
        if v not in results or results[v].get("meas2_R") is None:
            continue
        m = results[v]["meas2_R"]
        cos = m["cosine"]
        ab = m["acc_ablated"]
        ac = m["acc_reconstructed"]
        delta = (ac - ab) if (ac == ac and ab == ab) else float("nan")
        print(
            f"{v:<10}{_fmt(cos):>10}{m['acc_real']:>10.4f}"
            f"{ab:>10.4f}{_fmt(ac, w=11)}{_fmt(delta, w=12, p=4)}"
        )

    # ====================================================================
    # TABLE 3 — Causal coupling (noise on h_L)
    # ====================================================================
    print(f"\n{'=' * 90}")
    print("=== TABLE 3: Causal coupling — accuracy as noise added to h_L ===")
    print(f"{'=' * 90}")
    eps_used = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
    eps_hdr = "".join(f"  ε={e:>4.2f}" for e in eps_used)
    print(f"{'variant':<10}{eps_hdr}     IAS@0.5")
    print("-" * 90)
    for v in PRINT_ORDER:
        if v not in results:
            continue
        m = results[v].get("meas3")
        if m is None or not m["applicable"]:
            continue
        accs = m["accuracies"]
        accs_str = "".join(f"  {a:>6.4f}" for a in accs)
        ias = m["ias_at_0p5"]
        print(f"{v:<10}{accs_str}  {_fmt(ias, w=8, p=4)}")

    # ====================================================================
    # Hypothesis-relevant comparisons
    # ====================================================================
    print(f"\n\n{'=' * 80}")
    print("=== Hypothesis-relevant comparisons ===")
    print(f"{'=' * 80}")

    # Helper to look up a meas2_R value with a NaN fallback.
    def _m2(v: str, key: str) -> float:
        r = results.get(v)
        if r is None or r.get("meas2_R") is None:
            return float("nan")
        return r["meas2_R"].get(key, float("nan"))

    print("\n--- V3 vs B4 (★ central hypothesis test) ---")
    if "V3" in results and "B4" in results:
        d_cos = _m2("V3", "cosine") - _m2("B4", "cosine")
        d_acc = _m2("V3", "acc_reconstructed") - _m2(
            "B4", "acc_reconstructed"
        )
        d_task = (
            results["V3"]["task_acc_test"] - results["B4"]["task_acc_test"]
        )
        print(f"  cosine Δ        : {d_cos:+.4f}   (Scenario A 임계 ≥ +0.15)")
        print(f"  acc_recon Δ     : {d_acc:+.4f}   (Scenario A 임계 ≥ +0.05)")
        print(f"  task_acc Δ      : {d_task:+.4f}   (sanity, B4 ~비슷할 것)")
        for label, v in [("cosine", d_cos), ("acc_recon", d_acc)]:
            tag = (
                "✅ Scenario A pass"
                if (label == "cosine" and v >= 0.15)
                or (label == "acc_recon" and v >= 0.05)
                else "❌ below A"
                if (label == "cosine" and v < 0.0)
                or (label == "acc_recon" and v < 0.0)
                else "⚠ partial (B)"
            )
            print(f"    [{label}] {tag}")

    print("\n--- ACC ablation (V1 vs V2 vs V3) ---")
    for v in ("V1", "V2", "V3"):
        if v not in results:
            continue
        m = results[v].get("meas2_R")
        if m is None:
            continue
        print(
            f"  {v}: cos={_m2(v, 'cosine'):.4f}  "
            f"acc_real={_m2(v, 'acc_real'):.4f}  "
            f"acc_recon={_m2(v, 'acc_reconstructed'):.4f}  "
            f"Δ(rec-abl)={_m2(v, 'acc_reconstructed') - _m2(v, 'acc_ablated'):+.4f}"
        )

    print("\n--- D-21 baseline reminder ---")
    print(
        f"  r_random_baseline = {rb_mean:.4f}.  "
        f"Measurement #4 needs Day 7's 5-seed run; only #4-related output here."
    )

    # If any value is NaN/missing, warn loudly.
    n_nan = sum(
        1
        for v in results.values()
        if v.get("meas2_R") is not None
        and math.isnan(v["meas2_R"].get("acc_reconstructed", float("nan")))
        and v["meas2_R"].get("applicable_recon")
    )
    if n_nan > 0:
        print(f"\n[WARN] {n_nan} ACC-applicable variant(s) returned NaN acc_recon.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
