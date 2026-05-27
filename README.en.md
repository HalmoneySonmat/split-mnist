# SPLIT-MNIST

[🇰🇷 한국어](README.md) · 🇬🇧 English · [🇨🇳 中文](README.zh.md) · [🇯🇵 日本語](README.ja.md)

*A PoC verifying that when two small CNNs and a learnable artificial corpus callosum between them are co-trained from scratch, stimulating one half is enough to recover the other half's representation.*

After the neighboring project SPLIT-9 (a post-hoc adapter) ended on a negative — a structural ceiling on faithfulness — this PoC tests the hypothesis meant to break that ceiling: co-train two networks *from scratch* with a *learnable artificial corpus callosum (ACC)* between them, on the smallest toy (MNIST split left/right).
The hypothesized form (V2, with a decoupled reconstruction loss) recovers the other side to cosine 0.81 / classifier accuracy 0.83 under ablation — versus the post-hoc baseline (B4): cosine +0.16, acc_recon +0.11, both p<0.0001. 5 of 6 pre-registered thresholds pass.
Meanwhile the neuroscience-natural Hebbian term (V3) acts as *noise* in this ML implementation (−0.51 cosine vs V2) — itself a finding. A **positive** result that quantitatively shows the mechanism works in a toy.

**Headline:** V2 cosine 0.81 / acc_recon 0.83 · vs B4 +0.16 / +0.11 (p<0.0001) · position invariance r 0.46 vs random 0.75 · 5/6 pre-registered thresholds (Scenario A strong verification).

**[See the details → method · 4 measurements · tables (KO/EN)](https://halmoneysonmat.github.io/split-mnist/)**

---

### Research lineage

SPLIT-9's negative motivated this PoC, and this positive leads to the next step — growing two networks *together* from the start (co-developed twins).

[SPLIT-9](https://github.com/HalmoneySonmat/split-9) → **SPLIT-MNIST** → [SPLIT-MAZE](https://github.com/HalmoneySonmat/split-maze)

Reproduction, stack, and references are on the details page. Tests: `pytest tests/ -v` (~156 tests).

---

Claude was used to build this project.
