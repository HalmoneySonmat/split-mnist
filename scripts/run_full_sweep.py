#!/usr/bin/env python
"""Day 7 — full sweep: 5 seeds × 5 epochs × 7 variants + measurements #1–#4.

This is the **definitive PoC run**. After this completes you have:
  - mean ± std task accuracy per variant
  - mean ± std cross-activation (cosine, acc_recon) per variant
  - paired-bootstrap p-values for V2 vs B4 and V3 vs V2 (the two
    hypothesis-relevant comparisons)
  - position invariance r per variant vs random baseline
  - causal coupling curves per variant

Output:
  - Live progress to stdout
  - Aggregated results saved to runs/sweep_results.json
  - Console tables A/B/C/D + scenario verdict

Usage:
    python scripts/run_full_sweep.py
    python scripts/run_full_sweep.py --epochs 5 --seeds 42,43,44,45,46
    python scripts/run_full_sweep.py --variants V2,V3,B4   # quick subset
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from split_mnist.data import make_loaders
from split_mnist.evaluate import (
    measure_causal_coupling,
    measure_cross_activation,
    measure_position_invariance,
    measure_random_baseline_invariance,
)
from split_mnist.train import (
    TrainConfig,
    evaluate_left_only,
    train_one_run,
)


DEFAULT_VARIANTS = ("B1", "B2a", "B3", "B4", "V1", "V2", "V3")
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
PRINT_ORDER = ("B1", "B2a", "B2b", "B3", "B4", "V1", "V2", "V3")
ACC_LIKE = ("V1", "V2", "V3", "B4")  # variants with W_final


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEEDS),
    )
    p.add_argument(
        "--variants", type=str, default=",".join(DEFAULT_VARIANTS)
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--out", type=str, default="runs/sweep_results.json"
    )
    p.add_argument(
        "--no-measure-causal",
        action="store_true",
        help="Skip measurement #3 to save ~1min/seed.",
    )
    return p.parse_args()


# ----------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------


def _mean_std(xs: list[float]) -> tuple[float, float]:
    arr = np.array([x for x in xs if x == x], dtype=float)  # drop NaN
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)


def _paired_bootstrap_p(
    diffs: list[float], n_bootstrap: int = 10_000, seed: int = 0
) -> float:
    """Two-sided paired bootstrap p-value for H0: mean(diffs) = 0.

    Args:
        diffs: per-seed differences (e.g. [V2_seed_i - B4_seed_i for i]).
        n_bootstrap: resample count.

    Returns:
        Two-sided p-value.
    """
    arr = np.array(diffs, dtype=float)
    n = arr.size
    if n == 0 or np.any(np.isnan(arr)):
        return float("nan")
    rng = np.random.default_rng(seed)
    # Center under null (subtract observed mean); resample residuals.
    centered = arr - arr.mean()
    boot_means = rng.choice(centered, size=(n_bootstrap, n), replace=True).mean(axis=1)
    obs = arr.mean()
    # Two-sided p-value: how often does |boot mean| ≥ |obs|?
    return float((np.abs(boot_means) >= abs(obs)).mean())


def _to_jsonable(obj):
    """Recursively convert a value to JSON-serializable form."""
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "tolist"):  # numpy / torch
        return _to_jsonable(obj.tolist())
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> int:
    args = _parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(args.out)

    print(
        f"=== Day 7 full sweep: {len(variants)} variants × {len(seeds)} seeds"
        f" × {args.epochs} epochs on {device_str} ==="
    )
    print(f"variants: {variants}")
    print(f"seeds:    {seeds}\n")

    # ------------------------------------------------------------------
    # D-21 random baseline (one-time, no training)
    # ------------------------------------------------------------------
    print("--- D-21: random Procrustes baseline ---")
    rb = measure_random_baseline_invariance(hidden_dim=64, n_seeds=5)
    rb_mean = rb["procrustes_corr_mean"]
    print(f"r_random_baseline = {rb_mean:.4f}\n")

    # ------------------------------------------------------------------
    # Per-(seed, variant) training + measurement.
    # We use a SEPARATE loader per seed to ensure the train/val split is
    # seeded by the run seed (not a fixed seed-42 split). Loaders are
    # rebuilt outside the variant loop so all variants for one seed see
    # identical data.
    # ------------------------------------------------------------------
    # results[variant][seed] = dict
    results: dict[str, dict[int, dict]] = {v: {} for v in variants + ["B2b"]}
    # Per-(variant, seed) W_final tensors for measurement #4.
    W_per_seed: dict[str, list] = {v: [] for v in ACC_LIKE}

    overall_t0 = time.time()
    for seed in seeds:
        print(f"\n{'=' * 70}")
        print(f"=== SEED {seed} ===")
        print(f"{'=' * 70}")

        train_loader, val_loader, test_loader = make_loaders(
            root=args.data_root,
            batch_size=args.batch_size,
            seed=seed,
        )

        for variant in variants:
            print(f"\n[seed={seed}] training {variant}...")
            cfg = TrainConfig(
                variant=variant,
                seed=seed,
                n_epochs=args.epochs,
                batch_size=args.batch_size,
                log_every=99999,  # silence per-step logs
                device=device_str,
            )
            t0 = time.time()
            r = train_one_run(
                cfg,
                _data_loaders=(train_loader, val_loader, test_loader),
            )
            train_time = time.time() - t0

            # Measurement #2 (right_ablation only — symmetric direction is a
            # consistency check; for the headline numbers we use right).
            t0 = time.time()
            m2 = measure_cross_activation(
                r["model"], test_loader, variant,
                device=device_str, direction="right_ablation",
            )
            m3 = (
                measure_causal_coupling(
                    r["model"], test_loader, variant,
                    device=device_str,
                    epsilons=(0.0, 0.1, 0.25, 0.5, 1.0, 2.0),
                )
                if not args.no_measure_causal
                else None
            )
            meas_time = time.time() - t0

            results[variant][seed] = {
                "task_acc_test": r["best_test_acc"],
                "task_acc_val": r["best_val_acc"],
                "train_time_s": train_time,
                "meas_time_s": meas_time,
                "meas2_R": m2,
                "meas3": m3,
                "final_metrics": r["final_metrics"],
            }

            if variant in ACC_LIKE and r["W_final"] is not None:
                W_per_seed[variant].append(r["W_final"])

            # B2b derivation (B2a only)
            if variant == "B2a":
                b2b_acc = evaluate_left_only(
                    r["model"], test_loader, device=device_str
                )
                results["B2b"][seed] = {
                    "task_acc_test": b2b_acc,
                    "task_acc_val": float("nan"),
                    "train_time_s": 0.0,
                    "meas_time_s": 0.0,
                    "meas2_R": None,
                    "meas3": None,
                    "final_metrics": {},
                    "note": "shared B2a model, eval=left_only",
                }

            print(
                f"  done. test_acc={r['best_test_acc']:.4f} "
                f"train={train_time:.1f}s meas={meas_time:.1f}s"
            )

    overall_time = time.time() - overall_t0
    print(f"\n\nTotal sweep time: {overall_time / 60:.1f} min\n")

    # ------------------------------------------------------------------
    # Measurement #4 (position invariance, per variant with W_final)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}\n=== Measurement #4: position invariance ===\n{'=' * 70}")
    inv_results: dict[str, dict] = {}
    for variant in ACC_LIKE:
        Ws = W_per_seed.get(variant, [])
        if len(Ws) < 2:
            inv_results[variant] = {
                "n_seeds": len(Ws),
                "skipped": True,
            }
            continue
        out = measure_position_invariance(Ws)
        out["delta_vs_random"] = out["procrustes_corr_mean"] - rb_mean
        inv_results[variant] = out
        print(
            f"{variant}: r_trained = {out['procrustes_corr_mean']:.4f}  "
            f"(Δ vs random = {out['delta_vs_random']:+.4f})"
        )
    print(f"\n  random baseline = {rb_mean:.4f}")
    print(
        "  Scenario A threshold: r_trained ≤ random − 0.15  "
        f"(= {rb_mean - 0.15:.4f})"
    )

    # ------------------------------------------------------------------
    # Aggregate per-variant tables
    # ------------------------------------------------------------------
    print(f"\n\n{'=' * 80}")
    print("=== TABLE A: Task accuracy (mean ± std over seeds) ===")
    print(f"{'=' * 80}")
    print(f"{'variant':<10}{'val':>14}{'test':>14}  notes")
    print("-" * 80)
    for v in PRINT_ORDER:
        if v not in results or not results[v]:
            continue
        per_seed = list(results[v].values())
        val_m, val_s = _mean_std([r["task_acc_val"] for r in per_seed])
        test_m, test_s = _mean_std([r["task_acc_test"] for r in per_seed])
        notes = ""
        if v == "V3":
            gs = [
                r["final_metrics"].get("g")
                for r in per_seed
                if r["final_metrics"].get("g") is not None
            ]
            if gs:
                gm, gs_ = _mean_std(gs)
                notes = f"g={gm:+.4f}±{gs_:.4f}"
        if val_m == val_m:
            print(
                f"{v:<10}{val_m:>9.4f}±{val_s:.3f} "
                f"{test_m:>9.4f}±{test_s:.3f}  {notes}"
            )
        else:
            print(f"{v:<10}{'  --':>14}{test_m:>9.4f}±{test_s:.3f}  {notes}")

    # ------------------------------------------------------------------
    print(f"\n{'=' * 88}")
    print("=== TABLE B: Cross-activation faithfulness (right_ablation, mean ± std) ===")
    print(f"{'=' * 88}")
    print(
        f"{'variant':<10}{'cosine':>14}{'acc_real':>14}{'acc_abl':>14}"
        f"{'acc_recon':>14}{'Δ(rec-abl)':>14}"
    )
    print("-" * 88)
    for v in PRINT_ORDER:
        if v not in results or not results[v]:
            continue
        m2_list = [
            r["meas2_R"] for r in results[v].values() if r["meas2_R"] is not None
        ]
        if not m2_list:
            continue
        cos_m, cos_s = _mean_std([m["cosine"] for m in m2_list])
        ar_m, ar_s = _mean_std([m["acc_real"] for m in m2_list])
        ab_m, ab_s = _mean_std([m["acc_ablated"] for m in m2_list])
        ac_m, ac_s = _mean_std([m["acc_reconstructed"] for m in m2_list])
        d_m, d_s = _mean_std(
            [
                m["acc_reconstructed"] - m["acc_ablated"]
                for m in m2_list
                if m["acc_reconstructed"] == m["acc_reconstructed"]
            ]
        )
        cos_str = f"{cos_m:>9.4f}±{cos_s:.3f}" if cos_m == cos_m else f"{'  --':>14}"
        ac_str = f"{ac_m:>9.4f}±{ac_s:.3f}" if ac_m == ac_m else f"{'  --':>14}"
        d_str = f"{d_m:>+9.4f}±{d_s:.3f}" if d_m == d_m else f"{'  --':>14}"
        print(
            f"{v:<10}{cos_str}"
            f"{ar_m:>9.4f}±{ar_s:.3f}"
            f"{ab_m:>9.4f}±{ab_s:.3f}"
            f"{ac_str}{d_str}"
        )

    # ------------------------------------------------------------------
    print(f"\n{'=' * 88}")
    print("=== TABLE C: Position invariance (5 seeds, vs random baseline) ===")
    print(f"{'=' * 88}")
    print(f"{'variant':<10}{'r_trained':>14}{'Δ vs random':>16}  scenario")
    print("-" * 88)
    for v in ACC_LIKE:
        if v not in inv_results:
            continue
        ir = inv_results[v]
        if ir.get("skipped"):
            continue
        d = ir["delta_vs_random"]
        if d <= -0.15:
            sc = "✓ A (위치 무관 OK)"
        elif d >= 0.15:
            sc = "수렴 발견"
        else:
            sc = "B (모호 — random과 구별 안 됨)"
        print(
            f"{v:<10}{ir['procrustes_corr_mean']:>14.4f}{d:>+16.4f}  {sc}"
        )

    # ------------------------------------------------------------------
    # Hypothesis-relevant comparisons
    # ------------------------------------------------------------------
    print(f"\n\n{'=' * 80}")
    print("=== Hypothesis-relevant comparisons (paired bootstrap) ===")
    print(f"{'=' * 80}")

    def _get_per_seed(v: str, key: str) -> list[float]:
        out = []
        for s in seeds:
            r = results.get(v, {}).get(s)
            if r is None or r.get("meas2_R") is None:
                out.append(float("nan"))
                continue
            out.append(r["meas2_R"].get(key, float("nan")))
        return out

    if "V2" in variants and "B4" in variants:
        print("\n--- V2 vs B4 (★ central hypothesis: V2 = 본 가설 본형) ---")
        for key, threshold, label in [
            ("cosine", 0.15, "cosine"),
            ("acc_reconstructed", 0.05, "acc_recon"),
        ]:
            v2 = _get_per_seed("V2", key)
            b4 = _get_per_seed("B4", key)
            diffs = [v2[i] - b4[i] for i in range(len(seeds))]
            d_mean, d_std = _mean_std(diffs)
            p = _paired_bootstrap_p(diffs)
            tag = "✓" if d_mean >= threshold and p < 0.05 else (
                "△" if d_mean > 0 else "✗"
            )
            print(
                f"  {label:<14} Δ={d_mean:+.4f}±{d_std:.4f}  "
                f"p={p:.4f}  (Scenario A 임계 ≥ {threshold})  {tag}"
            )

    if "V3" in variants and "V2" in variants:
        print("\n--- V3 vs V2 (Hebbian 추가 효과 ablation) ---")
        for key, label in [("cosine", "cosine"), ("acc_reconstructed", "acc_recon")]:
            v3 = _get_per_seed("V3", key)
            v2 = _get_per_seed("V2", key)
            diffs = [v3[i] - v2[i] for i in range(len(seeds))]
            d_mean, d_std = _mean_std(diffs)
            p = _paired_bootstrap_p(diffs)
            print(
                f"  {label:<14} Δ={d_mean:+.4f}±{d_std:.4f}  "
                f"p={p:.4f}  (음수 = Day 6 D-20 재확정)"
            )

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "variants": variants,
            "seeds": seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": device_str,
        },
        "random_baseline": {
            "procrustes_corr_mean": rb_mean,
            "procrustes_corr_per_seed": rb["procrustes_corr_per_seed"],
        },
        "results_per_seed": _to_jsonable(
            {v: results.get(v, {}) for v in PRINT_ORDER if v in results}
        ),
        "position_invariance": _to_jsonable(inv_results),
        "total_time_s": overall_time,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n\nResults saved to: {out_path}")
    print(f"Total time: {overall_time / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
