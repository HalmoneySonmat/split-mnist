# SPLIT-MNIST

[🇰🇷 한국어](README.md) · 🇬🇧 English · [🇨🇳 中文](README.zh.md) · [🇯🇵 日本語](README.ja.md)

*A PoC verifying that when two small CNNs and an artificial corpus callosum
between them are co-trained from scratch, stimulating one half is enough to
recover the other half's representation.*

> **TL;DR.** The neighboring project [SPLIT-9](../split_brain_go) ended on
> a negative: a *post-hoc* adapter coupling a frozen Go-Net to a frozen LLM
> hits a structural ceiling on faithfulness — 95% of its loss reduction is
> a domain prior, only 5% is per-board signal, and 0/10 sampled outputs
> match the actor's actual move coordinate. This PoC tests whether the
> next hypothesis — *co-training + decoupled learning signals on the
> ACC* — works in the smallest, cleanest toy. We split MNIST 28×28 into
> left/right 14×28, run two small CNNs and an artificial corpus callosum
> (ACC) between them, and train *all of it* from scratch jointly. After
> 5 seeds × 5 epochs × 7 variants, the hypothesized form (V2) recovers
> the right hidden from the left to **cosine 0.81 and classifier accuracy
> 0.83** under right-side mean ablation. Versus the joint cross-attention
> baseline (B4): cosine **+0.16**, acc_recon **+0.11**, both with
> p<0.0001. Position invariance also holds (r=0.46 vs random 0.75). 5 of
> 6 pre-registered Scenario-A thresholds pass — **strong verification**.
> Meanwhile the Hebbian component, which felt natural from neuroscience,
> turns out to be *noise* in this ML implementation: V3 (Hebbian +
> reconstruction) loses cosine 0.51 to V2 (reconstruction only) with
> p<0.0001 — itself a finding.

---

## Why this exists

In the split-brain experiments of Sperry and Gazzaniga, severing the
corpus callosum produces a striking phenomenon: when an instruction is
shown only to the right hemisphere ("walk"), the patient stands and
walks. Asked *why*, the language-capable left hemisphere — which never
saw the instruction — confabulates a plausible reason. Not lying. The
patient genuinely *believes* their fabricated story. Gazzaniga called
this module the *left-hemisphere interpreter*: fluent, coherent, often
wrong.

Modern AI deliberately reproduces this architecture. Visual encoders,
robot policies, and decision models sit beside an LLM that "explains"
in natural language what is happening. RLHF chain-of-thought, vision-
language assistants, "explainable" RL agents — all the same pattern. And
in all of them, whether the LLM *actually translates* the upstream
signal or just produces statistically plausible text is, in general,
unverified.

The neighboring project SPLIT-9 attempted to study this directly with
9×9 Go + TinyLlama. The result: *cross-entropy training rationally
invests in priors weighted by token entropy*. The adapter learned was
~95% domain prior and only ~5% per-board signal. Of 10 qualitative
samples, 0 matched the actor's actual move coordinate. *A structural
ceiling for the post-hoc adapter approach.*

This PoC tests the next hypothesis to break that ceiling:

> If two networks are *co-trained from scratch* together with a
> learnable, position-invariant coactivation mapping (ACC) between them,
> stimulating one side should suffice to recover the other side's
> representation.

We test it in the smallest, cleanest toy: split MNIST.

---

## What we built

```
SPLIT-MNIST
├── MNIST 28×28
│       │
│       ├─ left  14×28 → left CNN ────┐
│       └─ right 14×28 → right CNN ───┤
│                                     │
│       ┌── ACC ─────────────────────┤   ← decoupled learning signal:
│       │   W ∈ ℝ^(64×64)            │     classification loss never
│       │   ĥ_R = h_L @ Wᵀ           │     touches W; ACC has its own
│       │   ĥ_L = h_R @ W            │     loss.
│       └── variant: V1 / V2 / V3 ───┘
│                                     │
│       cat[h_L, h_R] → classifier ───┘
│                                  │
│                                  ▼
│                          0–9 prediction (γ policy)
```

All components are **trained from scratch jointly**. Time and data are
shared; only the *gradient paths* split.

**Four ACC variants** (only the learning signal differs):
- **V1** Hebbian only — original neuroscience-faithful intuition ("fire together, wire together")
- **V2** ★ reconstruction only — backprop on a separate reconstruction loss; the *hypothesized form*
- **V3** Hebbian + reconstruction — both
- **B4** joint cross-attention — trained by the classification loss only (post-hoc adapter baseline; SPLIT-9 pattern reproduced in this toy)

Additional baselines: B1 (single CNN, ceiling), B2(a)/(b) (independent
classifiers / left-only at eval), B3 (raw concat). Seven+ variants.

**Data.** Standard MNIST 60k train (50k+10k val) + 10k test. Columns
0–13 vs 14–27, no overlap. Standard normalization.

**Training.** AdamW lr=1e-3, batch 128, 5 epochs, ~28s per run on RTX 3070
Ti. 35 runs total = 19.5 minutes.

---

## What we measured

### 1. Task accuracy (sanity)

After 5 epochs all variants land between 98.5% and 98.9% — within ~0.2pp
of the ceiling B1 (98.72%). **The ACC design barely affects classification
accuracy on this task** — it's measurement #2 below that *separates the
variants*.

### 2. Cross-activation faithfulness ★

We mean-ablate the right hidden, ask the ACC to reconstruct it from the
left, then compare the reconstruction to the true right hidden two ways:
cosine similarity, and classifier accuracy when the reconstruction is
substituted in.

| variant | cosine | acc_real | acc_ablated | acc_recon | Δ(rec−abl) |
|---|---:|---:|---:|---:|---:|
| **V2** ★ | **0.814 ± 0.007** | 0.985 | 0.756 ± 0.029 | **0.829 ± 0.043** | **+0.073** |
| B4 | 0.653 ± 0.026 | 0.986 | 0.740 ± 0.040 | 0.722 ± 0.039 | −0.018 |
| V3 | 0.303 ± 0.034 | 0.985 | 0.759 ± 0.037 | 0.681 ± 0.052 | −0.079 |
| V1 | 0.273 ± 0.107 | 0.985 | 0.737 ± 0.056 | 0.418 ± 0.080 | −0.319 |

**V2 is the only variant where `acc_recon > acc_ablated`** — its
reconstruction *actually helps* classification. Every other variant does
*worse than random ablation*.

Note especially **B4** with Δ = −0.018 — the joint-trained cross-attention
is *worse than random ablation* at recovery. This reproduces the
SPLIT-9 negative result inside this toy: post-hoc adapters get
classification right but cannot recover representation.

### 3. Causal coupling — toy limitation

We added Gaussian noise to the left hidden at ε ∈ {0, 0.1, 0.25, 0.5,
1.0, 2.0}. Across all variants, IAS@0.5 ≈ 0. **Split MNIST is too easy**:
the left half alone gets ≥95% classification, so the right half is
redundant, so noise on the left barely shifts the verdict. Measurement
#3 is not informative on this toy. A harder task is needed
(D-22 deferred).

### 4. Position invariance — D-21 random baseline

We Procrustes-aligned the 5-seed ACC W matrices and measured their mean
correlation r. Random isotropic 64×64 matrices already give r ≈ 0.75
(random matrix theory — D-21). Compared to that baseline:

| variant | r_trained | Δ vs random (0.7481) | verdict |
|---|---:|---:|---|
| **V2** | **0.4646** | **−0.2835** | ✓ A (clear position invariance) |
| V1 | 0.4105 | −0.3376 | ✓ A |
| V3 | 0.5600 | −0.1881 | ✓ A (marginal) |
| B4 | 0.4055 | −0.3426 | ✓ A |

V2 clears the 0.15 threshold by ~2×. **The 5 seeds learn *different*
pairing patterns — strong support for the position-invariance part of
the hypothesis.** Notably V3 has the *highest* r (0.56), i.e. the
*least* position-invariant — Hebbian pulls W toward seed-shared
statistics, which is the noise we documented elsewhere.

### Paired bootstrap — statistical clincher

```
V2 vs B4 (★ central hypothesis test):
  cosine     Δ = +0.1606 ± 0.0306   p = 0.0000   threshold ≥ 0.15  ✓
  acc_recon  Δ = +0.1066 ± 0.0303   p = 0.0000   threshold ≥ 0.05  ✓

V3 vs V2 (Hebbian-augmentation effect):
  cosine     Δ = −0.5111 ± 0.0325   p = 0.0000  ← Hebbian = noise, confirmed
  acc_recon  Δ = −0.1484 ± 0.0926   p = 0.0000
```

**Both Scenario-A thresholds passed at p = 0.0000.** All 5 seeds rank V2
above B4. The probability of this being noise is essentially zero.

---

## Verdict — Scenario A, strong verification

Out of 6 thresholds pre-registered in PLAN §12.A:

| measurement | threshold | result | passed |
|---|---|---|---|
| #1 V2 ≥ B1 − 5pp | 92.4% | 98.5% | ✓ |
| #2 V2 cosine ≥ 0.7 | 0.7 | 0.81 | ✓ |
| #2 V2 vs B4 cosine Δ ≥ 0.15 | 0.15 | +0.16 (p<0.0001) | ✓ |
| #2 V2 vs B4 acc_recon Δ ≥ 5pp | 0.05 | +0.11 (p<0.0001) | ✓ |
| #3 V2 causal coupling | 5pp | toy limit | ⚠ |
| #4 V2 r ≤ random − 0.15 | ≤ 0.60 | 0.46 | ✓ |

**5 of 6 pass.** Only measurement #3 is weak, due to the toy's
limitation acknowledged *before* seeing the result (PLAN §18.5).

---

## What this means

**Co-training two networks with a learnable corpus callosum between them
can solve the cross-activation faithfulness problem that post-hoc adapters
hit a ceiling on (the SPLIT-9 pattern).** This PoC quantitatively
demonstrates that *in principle*, in the smallest possible toy.

The crux is **decoupling the learning signals** (the γ policy): if the
ACC is *not* directly backpropagated into by the classification loss but
instead has its own reconstruction loss, the ACC actually learns the
mapping. The joint cross-attention baseline (B4) gets pulled by the
classification loss into "whatever helps classification" — which, it
turns out, is *not the same as recovering the other side's
representation*. This pinpoints the cause of SPLIT-9's negative result.

A second finding: **Hebbian learning is a poor fit in this ML
implementation.** In V3, W_hebbian accumulates an arbitrary outer-product
statistic, and W_learned spends its capacity *cancelling that statistic*
rather than learning the true mapping (η/W_max sweeps confirm this is a
*structural* limit, not a hyperparameter issue). V3 loses cosine 0.51 to
V2. A neuroscience-faithful intuition turns out to *hurt* in ML —
**an artificial corpus callosum needs no Hebbian; a separated
reconstruction loss with γ-policy is sufficient**.

This conflicts with the direction taken by recent ML work that imports
Hebbian-style updates wholesale (Hybrid SOM 2022, HeLa-Mem 2025, Hebbian
Fast Weights in ViT 2025). Our finding suggests the *co-training
environment + reconstruction loss decoupled from classification* is what
matters; Hebbian may be an option, not a foundation.

---

## Limits (PoC scope)

Four reasons to be careful about generalizing:

1. **MNIST.** The two modalities are homogeneous (both small CNNs, both
   image halves). Truly heterogeneous setups (e.g. a decision model and
   an LLM) may behave differently.
2. **Task too easy.** Left half alone gets ≥95% — measurement #3
   (causal coupling) is not informative.
3. **No "interpretation" tested.** This PoC verifies *numerical
   recovery*; SPLIT-9's natural-language faithfulness is a separate
   question.
4. **One-way only.** The full form of the user's hypothesis (bidirectional
   corpus callosum — the LLM's explanation feeding back into the next
   decision) is unverified.

---

## Future work — SPLIT-9 reboot

Given that the mechanism works in a toy, the natural next steps:

1. **Apply V2-style ACC inside SPLIT-9** — co-train a 9×9 Go-Net and a
   frozen LLM through a V2 ACC. Does SPLIT-9's negative result (95%
   prior, 5% board-specific) dissolve?
2. **Bidirectional corpus callosum** — let the LLM's explanation
   influence Go-Net's next decision. The full form of the user's
   hypothesis (split-brain → normal brain).
3. **D-22**: a harder task where measurement #3 is informative.
4. **D-23**: V3' "Hebbian opt-in" (`W = tanh(α)·W_h + W_l`) — does
   Hebbian help under any condition?

This PoC is the base camp on that route — quantitative confirmation
that the mechanism works in toy form.

---

## Reproducing

```bash
# Environment (validated on RTX 3070 Ti / WSL2 Ubuntu, single 8 GB GPU)
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Tests
pytest tests/ -v   # 156 tests

# Day 7 full sweep — 5 seeds × 5 epochs × 7 variants (~20 min)
python scripts/run_full_sweep.py
```

Results saved to `runs/sweep_results.json`. Console prints four tables
plus paired-bootstrap p-values.

Requirements: Python 3.10–3.12, PyTorch 2.3+, CUDA 12.1, NumPy, SciPy.

---

## Stack

Python 3.10–3.12 · PyTorch 2.3+ · NumPy · SciPy · pytest · single 8 GB
consumer GPU. CUDA 12.1.

---

## References

* Alayrac et al., *Flamingo: a Visual Language Model for Few-Shot Learning*, NeurIPS 2022.
* Haxby et al., *Hyperalignment: Modeling shared information encoded in idiosyncratic cortical topographies*, eLife 2020.
* Lambon Ralph et al., *The Roles of Left Versus Right Anterior Temporal Lobes in Semantic Memory*, Cerebral Cortex 2018.
* Koganemaru et al., paired associative stimulation and interhemispheric Hebbian PAS work.
* Innocenti & Price, *Exuberance in the development of cortical networks*, Nature Reviews Neuroscience 2005.
* Gazzaniga, *The Bisected Brain*, Appleton-Century-Crofts, 1970.
* Gazzaniga, *The Consciousness Instinct*, Farrar Straus Giroux, 2018.
* Turpin et al., *Language Models Don't Always Say What They Think*, NeurIPS 2023.
* Atanasova et al., *Faithfulness Tests for Natural Language Explanations*, ACL 2023.
* Conmy et al., *Towards Automated Circuit Discovery for Mechanistic Interpretability*, NeurIPS 2023.
* (neighboring SPLIT-9 — post-hoc adapter negative result)

---

## Status

PoC complete. v0.3-day7-scenarioA. Strong Scenario-A verification.
Workshop-paper writeup is feasible. The natural next step is the
SPLIT-9 reboot (apply V2-style ACC to 9×9 Go + LLM).

---

## License

Apache License 2.0. See [LICENSE](LICENSE).

```
Copyright 2026 namdo

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

---

*"When two networks grow together, grow the bridge between them
together too. Give the bridge its own work to do. Then what one side
sees, the other side wakes to."* — the conclusion of this PoC, in one
line.

---

*This was a fun project.*
