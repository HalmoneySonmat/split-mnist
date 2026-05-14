# SPLIT-MNIST

[🇰🇷 한국어](README.md) · [🇬🇧 English](README.en.md) · [🇨🇳 中文](README.zh.md) · 🇯🇵 日本語

*MNISTを左右に分割し、二つの小さなCNNとその間の学習可能な人工脳梁を*
*同時に*訓練したとき、片側の刺激だけで反対側の表現を本当に復元できる
かを検証したPoC。*

> **TL;DR.** 隣接プロジェクト [SPLIT-9](../split_brain_go) は、
> *post-hoc アダプター* が自然言語説明の忠実性に構造的天井を持つという
> negative resultで終わった。本PoCはその限界を解く *共同学習 + 学習信号
> の分離されたACC* 仮説が動作するかを最小のtoyで検証する。MNIST 28×28を
> 左右14×28に分割し、二つの小さなCNNと間の人工脳梁(ACC)を*すべて
> from scratchで一緒に*訓練。5 seed × 5 epoch × 7 variant のフル
> スイープ結果、本仮説の本形(V2)は片側hidden の mean ablation 後でも
> **cosine 0.81、分類精度0.83** まで反対側を復元した。joint
> cross-attention ベースライン (B4) 比 cosine **+0.16**、acc_recon
> **+0.11** (両方 p<0.0001)。位置不変性も確認 (r=0.46 vs random 0.75)。
> 事前登録した6つの閾値のうち5つが通過 — **Scenario A 強い検証**。
> 一方ユーザー直観に忠実なHebbianコンポーネント (V3) は ML 実装で
> W_learned とリソース競合し*ノイズ*として作用 (V3 vs V2 cosine
> −0.51, p<0.0001) — これ自体が学術的発見。

---

## なぜ作ったか

SperryとGazzanigaの分離脳実験 — 左右半球をつなぐ脳梁を切断した患者に、
右半球だけが見える位置で「歩きなさい」とカードを示すと、患者は立ち
上がって歩く。「なぜ歩くの?」と尋ねると、言語能力を持つ左半球は*真の
理由*を見ていないにもかかわらず、*もっともらしい話*を作り出す — 嘘
ではなく本当に*そうだと信じている*。Gazzanigaが*左半球通訳*と呼んだ
モジュールである。流暢で、一貫していて、しばしば間違う。

現代AIはこの構造を意図的に再現する。視覚エンコーダー、ロボット政策、
意思決定モデルなどの非言語ネットワークの隣にLLMを配置し、「今何が
起きているか」を自然言語で説明させる。RLHFのchain-of-thought、
ビジョン・ランゲージアシスタント、「説明可能な」RLエージェント — すべ
て同じパターン。しかしLLMが*本当に上流の信号を翻訳*しているのか、
あるいは*統計的にもっともらしいテキストを吐いているだけ*なのかは、
通常検証されない。

隣接プロジェクトSPLIT-9は9×9碁 + TinyLlamaでこの問いに直接挑んだ。
結果は、*クロスエントロピー学習はトークンエントロピーに沿って prior
にリソースを投資する*。アダプターが学習したものの95%はドメイン prior、
5%だけが盤面ごとの信号だった。定性的サンプル10個のうち0個が actor の
実際の着手座標と一致。*post-hoc アダプターアプローチの構造的天井*。

本PoCはその天井を破る次の仮説を検証する:

> 二つのニューラルネットワークを*同時にfrom scratchで*訓練し、その間
> に*位置不変な共活性化マッピング*を学習する単位(ACC)を置けば、片側の
> 刺激だけで反対側の表現を復元できる。

これを最小で最もクリーンなtoy — 分離MNIST — で定量的に検証する。

---

## 何を作ったか

```
SPLIT-MNIST
├── MNIST 28×28
│       │
│       ├─ 左 14×28 → 左 CNN ──────────┐
│       └─ 右 14×28 → 右 CNN ──────────┤
│                                       │
│       ┌── 人工脳梁 (ACC) ────────────┤   ← 学習信号を分離:
│       │   W ∈ ℝ^(64×64)              │     分類 loss は ACC を
│       │   ĥ_R = h_L @ Wᵀ              │     直接backpropしない
│       │   ĥ_L = h_R @ W               │     ACC は自分の loss で学習
│       └── 変形: V1 / V2 / V3 ────────┘
│                                       │
│       cat[h_L, h_R] → 分類器 ──────────┘
│                                  │
│                                  ▼
│                            0~9 分類 (γ ポリシー)
```

すべてのコンポーネントが**from scratchの共同学習**。学習の*時間とデータ*
は共有、学習の *gradient 経路*のみ分離。

**4つのACC変形** (学習信号の違いだけ):
- **V1** Hebbian only — 元のニューロサイエンス的直観 (「一緒に発火すれば結合」)
- **V2** ★ 再構成 only — backprop で片側→反対側のマッピングを学習 (本仮説の本形)
- **V3** Hebbian + 再構成 — 両方
- **B4** joint cross-attention — 分類 loss のみで学習 (post-hoc アダプター ベースライン、SPLIT-9パターンの本toy再現)

追加ベースライン: B1 (統合CNN, 天井)、B2(a)(b) (independent / left-only)、
B3 (raw concat)。計7+ variant。

**データ.** 標準MNIST 60k train (50k+10k val) + 10k test。0–13列 vs
14–27列、重複なし。標準正規化のみ。

**学習.** AdamW lr=1e-3、batch 128、5 epoch、RTX 3070 Tiで約28秒/run。
35 run = 19.5分。

---

## 何を測定したか

### 1. Task accuracy (sanity)

5 epoch後、すべての variant が 98.5~98.9% に収まり、天井 B1 (98.72%)
にほぼ追いつく。**分類自体ではACC設計の差はほぼ見えない** —
測定 #2 が ACC variant を *識別する決定的なツール*。

### 2. Cross-activation faithfulness ★

右hidden を mean ablation。ACC が左 hidden だけから右 hidden を推定 →
本物の右 hidden と cosine 比較 + 推定値を分類器に通して精度比較。

| variant | cosine | acc_real | acc_ablated | acc_recon | Δ(rec−abl) |
|---|---:|---:|---:|---:|---:|
| **V2** ★ | **0.814 ± 0.007** | 0.985 | 0.756 ± 0.029 | **0.829 ± 0.043** | **+0.073** |
| B4 | 0.653 ± 0.026 | 0.986 | 0.740 ± 0.040 | 0.722 ± 0.039 | −0.018 |
| V3 | 0.303 ± 0.034 | 0.985 | 0.759 ± 0.037 | 0.681 ± 0.052 | −0.079 |
| V1 | 0.273 ± 0.107 | 0.985 | 0.737 ± 0.056 | 0.418 ± 0.080 | −0.319 |

**V2のみ `acc_recon > acc_ablated`** — ACC の復元が実際に分類を助ける。
他のすべての variant は random ablation よりも復元できない。

特に **B4** の Δ が負 (−0.018) — *joint 学習の cross-attention が
random よりも復元できない*。SPLIT-9 negative パターンの本 toy 再現。
「post-hoc adapter は分類はできるが復元できない」が統計的に確認。

### 3. Causal coupling — toyの限界

左 hidden に ε ∈ {0, 0.1, 0.25, 0.5, 1.0, 2.0} のガウスノイズを加えた。
すべての variant で IAS@0.5 ≈ 0。**MNIST 分離脳が易しすぎる** (左
14×28 だけで 95%+ の分類が可能 → 右は冗長 → 左ノイズも影響なし)。
測定 #3 は本 toy では情報的でない。より難しい task が必要 (D-22 deferred)。

### 4. Position invariance — D-21 random baseline

5 seedの ACC W を Procrustes 整列して平均相関 r を測定。64×64の random
isotropic 行列はそれ自体で r ≈ 0.75 (random matrix theory)。学習された
W はどうか:

| variant | r_trained | Δ vs random (0.7481) | 判定 |
|---|---:|---:|---|
| **V2** | **0.4646** | **−0.2835** | ✓ A (位置不変、強い) |
| V1 | 0.4105 | −0.3376 | ✓ A |
| V3 | 0.5600 | −0.1881 | ✓ A (わずか) |
| B4 | 0.4055 | −0.3426 | ✓ A |

V2が閾値0.15の約2倍を通過。**5 seed が*互いに異なる対応パターン*を
学習 — 位置不変仮説の強い検証**。興味深いことに V3 が最も*位置不変
ではない* (Hebbian が seed 間で類似した統計に収束しようとする = Hebbian
ノイズの正体)。

### Paired bootstrap — 統計的決定打

```
V2 vs B4 (★ 中心仮説テスト):
  cosine     Δ = +0.1606 ± 0.0306   p = 0.0000   閾値 ≥ 0.15  ✓
  acc_recon  Δ = +0.1066 ± 0.0303   p = 0.0000   閾値 ≥ 0.05  ✓

V3 vs V2 (Hebbian 追加効果):
  cosine     Δ = −0.5111 ± 0.0325   p = 0.0000  ← Hebbian ノイズ統計確定
  acc_recon  Δ = −0.1484 ± 0.0926   p = 0.0000
```

**両方とも Scenario A 閾値通過 + p = 0.0000.** 5 seed すべてで一貫して
V2 が優位 — 偶然の確率ほぼ0。

---

## シナリオ判定 — Scenario A 強い検証

PLAN §12 で事前登録した6閾値:

| 測定 | 閾値 | 結果 | 通過 |
|---|---|---|---|
| #1 V2 ≥ B1 − 5%p | 92.4% | 98.5% | ✓ |
| #2 V2 cosine ≥ 0.7 | 0.7 | 0.81 | ✓ |
| #2 V2 vs B4 cosine Δ ≥ 0.15 | 0.15 | +0.16 (p<0.0001) | ✓ |
| #2 V2 vs B4 acc_recon Δ ≥ 5%p | 0.05 | +0.11 (p<0.0001) | ✓ |
| #3 V2 因果結合 | 5%p | toy 限界 | ⚠ |
| #4 V2 r ≤ random − 0.15 | ≤ 0.60 | 0.46 | ✓ |

**5/6 通過**。測定 #3 のみ toy 限界で弱い (PLAN §18.5 で結果を見る*前*
に認められた)。

---

## これは何を意味するか

**二つのニューラルネットワークを共同学習させ、その間に*学習可能な
人工脳梁*を置けば、post-hoc アダプター (SPLIT-9 パターン) が解けなかった
cross-activation faithfulness 問題を解決できる。** 本PoCはその*原理的
可能性*を最小のtoyで定量的に立証。

核心は **学習信号の分離** (γ ポリシー): ACC が分類 loss から直接
backprop を受けず、自分の*再構成 loss* で学習すれば、ACC が真のマッピング
を学習する。joint cross-attention (B4) は分類 loss に引かれて*分類だけ
うまくいくマッピング*を学習 — 結果として random ablation よりも復元
できない。SPLIT-9 negative の*原因*が本 toy で確認された。

もう一つ — **Hebbian は ML 実装には不適合**。V3 では W_hebbian が任意
の outer product 統計に蓄積し、W_learned が*そのノイズの相殺*にリソース
を奪われる (η/W_max sweep で構造的限界を確定)。V3 は V2 より cosine 0.51
損失。ニューロサイエンス的に忠実な直観が ML では*逆効果* — **人工脳梁
は Hebbian なしで分離された再構成 loss + γ ポリシーだけで十分**。

この発見は Hebbian を直接導入した最近の ML 研究 (Hybrid SOM 2022,
HeLa-Mem 2025, Hebbian Fast Weights in ViT 2025) とは異なる結論。本
PoC は「共同学習環境 + 分類 loss から分離された再構成 loss」の組み合わせ
が核心であり、Hebbian は選択肢であって基盤ではない可能性を示唆する。

---

## 限界 (PoC の範囲)

本結果の一般化には注意すべき4点:

1. **MNIST**: 二つの modality が同質的 (両方とも小さな CNN、両方とも
   画像の半分)。真に異質な環境 (例: 意思決定モデル + LLM) では動作が
   異なる可能性。
2. **task が易しすぎる**: 左だけで 95%+ 分類可能。測定 #3 (因果結合)
   が情報的でない。
3. **「解釈」未検証**: 本 PoC は*数値的復元*のみ確認。SPLIT-9 の*自然
   言語説明の忠実性*は別の問題。
4. **片方向のみ**: ユーザー仮説の full form (双方向脳梁 — LLM の説明が
   次の意思決定に影響) は未検証。

---

## 今後の研究 — SPLIT-9 reboot

本 PoC でメカニズムが toy で動作することが示されたので、次の自然な
ステップ:

1. **SPLIT-9環境にV2形式のACCを適用** — 9×9 Go-Net + 凍結 LLM の間に
   *共同学習される V2 ACC*。SPLIT-9 の negative result (95% prior、
   5% 盤面別信号) が*解消される*か。
2. **双方向脳梁** — LLMの説明が Go-Net の次の意思決定に影響。ユーザー
   仮説の full form。分離脳 → 通常の脳への移行。
3. **D-22**: 測定 #3 が情報的なより難しい toy/task。
4. **D-23**: V3' "Hebbian opt-in" (`W = tanh(α)·W_h + W_l`) — Hebbian
   が特定の条件で役立つか。

本 PoC はその道のベースキャンプ — メカニズムが toy で動作することの
定量的確認。

---

## 再現方法

```bash
# 環境 (RTX 3070 Ti / WSL2 Ubuntu で検証、8 GB GPU 1台)
python -m venv .venv && source .venv/bin/activate
pip install -e .

# テスト
pytest tests/ -v   # 156件

# Day 7 full sweep — 5 seed × 5 epoch × 7 variant (~20分)
python scripts/run_full_sweep.py
```

結果は `runs/sweep_results.json` に保存。コンソールには表4つ +
paired bootstrap。

要件: Python 3.10–3.12, PyTorch 2.3+, CUDA 12.1, NumPy, SciPy。

---

## スタック

Python 3.10–3.12 · PyTorch 2.3+ · NumPy · SciPy · pytest · 8 GB
コンシューマー GPU 1台。CUDA 12.1。

---

## 参考文献

* Alayrac et al., *Flamingo: a Visual Language Model for Few-Shot Learning*, NeurIPS 2022.
* Haxby et al., *Hyperalignment: Modeling shared information encoded in idiosyncratic cortical topographies*, eLife 2020.
* Lambon Ralph et al., *The Roles of Left Versus Right Anterior Temporal Lobes in Semantic Memory*, Cerebral Cortex 2018.
* Koganemaru et al., paired associative stimulation と interhemispheric Hebbian PAS。
* Innocenti & Price, *Exuberance in the development of cortical networks*, Nature Reviews Neuroscience 2005.
* Gazzaniga, *The Bisected Brain*, Appleton-Century-Crofts, 1970.
* Gazzaniga, *The Consciousness Instinct*, Farrar Straus Giroux, 2018.
* Turpin et al., *Language Models Don't Always Say What They Think*, NeurIPS 2023.
* Atanasova et al., *Faithfulness Tests for Natural Language Explanations*, ACL 2023.
* Conmy et al., *Towards Automated Circuit Discovery for Mechanistic Interpretability*, NeurIPS 2023.
* (隣接フォルダ SPLIT-9 — post-hoc adapter negative result)

---

## ステータス

PoC 完結。v0.3-day7-scenarioA。Scenario A 強い検証。ワークショップ
short paper 作成可能。次の自然なステップは SPLIT-9 reboot (9×9 碁 + LLM
に V2 形式 ACC を適用)。

---

## ライセンス

Apache License 2.0。[LICENSE](LICENSE) を参照。

```
Copyright 2026 namdo

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

---

*「二つのネットワークを同時に育てるとき、その間の橋も同時に育てよ。
橋には自分の仕事をさせよ。そうすれば片方が見たものを、もう片方が
目覚める。」* — 本 PoC の結論を一文で。

---

*面白いプロジェクトだった。*
