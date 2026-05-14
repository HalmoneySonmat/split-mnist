# SPLIT-MNIST

[🇰🇷 한국어](README.md) · [🇬🇧 English](README.en.md) · 🇨🇳 中文 · [🇯🇵 日本語](README.ja.md)

*把 MNIST 切成左右两半,让两个小 CNN 和它们之间的人工胼胝体*同时*训练,*
*验证仅凭一侧的刺激能否真正恢复另一侧的表征。*

> **TL;DR.** 邻接项目 [SPLIT-9](../split_brain_go) 以 negative result 收尾:
> *post-hoc 适配器*在自然语言解释忠实性上撞到结构性天花板 —— 学到的
> 95% 是领域 prior,只有 5% 是棋局相关信号,10 个定性样本中 0 个匹配
> actor 实际着子坐标。本 PoC 用最小最干净的 toy 检验下一个假设:
> *共同训练 + 学习信号解耦的 ACC* 是否成立。把 MNIST 28×28 切成左右
> 14×28,两个小 CNN 和其间的人工胼胝体(ACC)*全部 from scratch 一起*
> 训练。5 seed × 5 epoch × 7 variant 全扫描后,本假设的本形(V2)在
> 对右侧 hidden 进行 mean ablation 之后,仍能将其恢复至 **cosine 0.81、
> 分类准确率 0.83**。相比 joint cross-attention 基线(B4):cosine
> **+0.16**、acc_recon **+0.11**(都 p<0.0001)。位置不变性也成立
> (r=0.46 vs random 0.75)。预注册的 6 个阈值通过 5 个 —— **Scenario A
> 强验证**。另一方面,神经科学直观上忠实的 Hebbian 组件(V3)在 ML 实现
> 中与 W_learned 争资源,作为*噪声*起作用(V3 vs V2 cosine −0.51,
> p<0.0001)—— 这本身就是一项学术发现。

---

## 为什么做这个

Sperry 与 Gazzaniga 的裂脑实验中,切断左右半球之间的胼胝体的患者会
出现一种引人注目的现象:把指令(比如「走」)只展示在右半球可见的位置,
患者就站起来走。被问到*为什么*时,具有语言能力的左半球 —— 它从未看到
那条指令 —— 会编造一个看似合理的理由。不是说谎。患者真心*相信*自己
编造的故事。Gazzaniga 把这个模块称作*左半球解释器*:流畅、连贯、
经常出错。

现代 AI 在故意复制这种结构。视觉编码器、机器人策略、决策模型这些
非语言网络的旁边放一个 LLM,让它用自然语言「解释」当下发生了什么。
RLHF chain-of-thought、视觉-语言助手、「可解释」的 RL 智能体 ——
都是同一个模式。在所有这些设置中,LLM 究竟是*真的在翻译*上游信号,
还是只在产生统计上看似合理的文本,通常未被检验。

邻接项目 SPLIT-9 试图用 9×9 围棋 + TinyLlama 直接研究这一点。结果是:
*交叉熵训练按照 token 熵的分布合理地把资源投到 prior 上*。学到的
适配器中约 95% 是领域 prior,只有约 5% 是棋局相关信号。10 个定性样本
中 0 个匹配 actor 实际着子坐标。*post-hoc 适配器路线的结构性天花板。*

本 PoC 检验打破这个天花板的下一个假设:

> 如果把两个网络*同时 from scratch 共同训练*,在它们之间放一个学习
> *位置无关的共激活映射*的单元(ACC),那么仅凭一侧的刺激就足以恢复
> 另一侧的表征。

我们在最小最干净的 toy —— 切分 MNIST —— 上做定量检验。

---

## 我们做了什么

```
SPLIT-MNIST
├── MNIST 28×28
│       │
│       ├─ 左 14×28 → 左 CNN ─────────┐
│       └─ 右 14×28 → 右 CNN ─────────┤
│                                     │
│       ┌── 人工胼胝体 (ACC) ──────────┤   ← 学习信号解耦:
│       │   W ∈ ℝ^(64×64)             │     分类 loss 不直接经过 ACC
│       │   ĥ_R = h_L @ Wᵀ            │     ACC 用自己的 loss 训练
│       │   ĥ_L = h_R @ W             │
│       └── 变形: V1 / V2 / V3 ───────┘
│                                     │
│       cat[h_L, h_R] → 分类器 ────────┘
│                                  │
│                                  ▼
│                          0–9 分类 (γ 策略)
```

所有组件**from scratch 共同训练**。训练的*时间和数据*共享,只有
*梯度路径*分离。

**4 种 ACC 变形**(只有学习信号不同):
- **V1** 仅 Hebbian —— 神经科学的原始直观(「同时激发就连接」)
- **V2** ★ 仅重构 —— 用 backprop 在单独的重构 loss 上训练映射;*本假设的本形*
- **V3** Hebbian + 重构 —— 两者结合
- **B4** joint cross-attention —— 仅靠分类 loss 训练(post-hoc 适配器
  基线;在本 toy 中复现 SPLIT-9 模式)

附加基线:B1(单一 CNN,天花板)、B2(a)/(b)(独立分类器 / 仅左侧)、
B3(原始 concat)。共 7+ 种变形。

**数据.** 标准 MNIST 60k 训练(50k+10k 验证)+ 10k 测试。0–13 列
vs 14–27 列,无重叠。仅标准化。

**训练.** AdamW lr=1e-3,batch 128,5 epoch,RTX 3070 Ti 上每次约 28 秒。
共 35 次 = 19.5 分钟。

---

## 我们测了什么

### 1. Task accuracy(sanity)

5 epoch 后所有变形落在 98.5%~98.9% 之间 —— 与天花板 B1(98.72%)
相差约 0.2pp 之内。**ACC 设计在分类精度上几乎不起作用** —— 真正*区分
变形*的是下面的测量 #2。

### 2. Cross-activation faithfulness ★

将右侧 hidden 做 mean ablation,让 ACC 仅凭左侧 hidden 推测右侧
hidden。把推测结果与真实右侧 hidden 做 cosine 比较 + 把推测结果代入
分类器测准确率。

| variant | cosine | acc_real | acc_ablated | acc_recon | Δ(rec−abl) |
|---|---:|---:|---:|---:|---:|
| **V2** ★ | **0.814 ± 0.007** | 0.985 | 0.756 ± 0.029 | **0.829 ± 0.043** | **+0.073** |
| B4 | 0.653 ± 0.026 | 0.986 | 0.740 ± 0.040 | 0.722 ± 0.039 | −0.018 |
| V3 | 0.303 ± 0.034 | 0.985 | 0.759 ± 0.037 | 0.681 ± 0.052 | −0.079 |
| V1 | 0.273 ± 0.107 | 0.985 | 0.737 ± 0.056 | 0.418 ± 0.080 | −0.319 |

**只有 V2 的 `acc_recon > acc_ablated`** —— ACC 的恢复值真的*帮到分类*。
其他所有变形都比 random ablation 还差。

特别是 **B4** 的 Δ 为负(−0.018)—— *joint 训练的 cross-attention
比 random ablation 还差*。SPLIT-9 negative 模式在本 toy 中的复现:
「post-hoc 适配器只擅长分类,不会恢复表征」得到统计确认。

### 3. Causal coupling —— toy 限制

向左侧 hidden 加入 ε ∈ {0, 0.1, 0.25, 0.5, 1.0, 2.0} 的高斯噪声。
所有变形 IAS@0.5 ≈ 0。**MNIST 裂脑太简单**:仅靠左 14×28 就能达到
≥95% 分类率 —— 右侧冗余 —— 左侧噪声也几乎不影响。测量 #3 在本 toy
上没有信息量。需要更难的 task(D-22 deferred)。

### 4. Position invariance —— D-21 random baseline

将 5 seed 的 ACC W 矩阵 Procrustes 对齐,测量平均相关 r。64×64 random
isotropic 矩阵本身的 r ≈ 0.75(random matrix theory —— D-21)。在
此基线上,训练得到的 W 表现如何:

| variant | r_trained | Δ vs random (0.7481) | 判定 |
|---|---:|---:|---|
| **V2** | **0.4646** | **−0.2835** | ✓ A(位置无关性强) |
| V1 | 0.4105 | −0.3376 | ✓ A |
| V3 | 0.5600 | −0.1881 | ✓ A(临界) |
| B4 | 0.4055 | −0.3426 | ✓ A |

V2 大约以阈值 0.15 的 2 倍通过。**5 seed 学到了*互不相同的配对模式*——
位置无关性假设的强验证**。值得注意的是 V3 的 r 最高(0.56),即*最不
位置无关*—— Hebbian 把 W 拉向 seed 间共享的统计,这正是我们记录的噪声。

### Paired bootstrap —— 统计决断

```
V2 vs B4(★ 中心假设检验):
  cosine     Δ = +0.1606 ± 0.0306   p = 0.0000   阈值 ≥ 0.15  ✓
  acc_recon  Δ = +0.1066 ± 0.0303   p = 0.0000   阈值 ≥ 0.05  ✓

V3 vs V2(Hebbian 增量效应):
  cosine     Δ = −0.5111 ± 0.0325   p = 0.0000  ← Hebbian 噪声统计确认
  acc_recon  Δ = −0.1484 ± 0.0926   p = 0.0000
```

**两个 Scenario-A 阈值都通过,p = 0.0000.** 5 seed 全部一致地把 V2
排在 B4 之上 —— 偶然性几乎为零。

---

## 判定 —— Scenario A 强验证

PLAN §12.A 中预注册的 6 个阈值:

| 测量 | 阈值 | 结果 | 通过 |
|---|---|---|---|
| #1 V2 ≥ B1 − 5pp | 92.4% | 98.5% | ✓ |
| #2 V2 cosine ≥ 0.7 | 0.7 | 0.81 | ✓ |
| #2 V2 vs B4 cosine Δ ≥ 0.15 | 0.15 | +0.16(p<0.0001) | ✓ |
| #2 V2 vs B4 acc_recon Δ ≥ 5pp | 0.05 | +0.11(p<0.0001) | ✓ |
| #3 V2 因果耦合 | 5pp | toy 限制 | ⚠ |
| #4 V2 r ≤ random − 0.15 | ≤ 0.60 | 0.46 | ✓ |

**6 个中 5 个通过**。只有测量 #3 弱,这是因为 toy 自身的局限,且在
*看到结果之前*已被 PLAN §18.5 承认。

---

## 这意味着什么

**把两个网络共同训练并在它们之间放一个可学习的人工胼胝体,可以解决
post-hoc 适配器(SPLIT-9 模式)所撞到天花板的 cross-activation
faithfulness 问题。** 本 PoC 在最小可能的 toy 中*原理上*定量证明了
这一点。

关键在于**学习信号的解耦**(γ 策略):如果 ACC 不被分类 loss 直接
backprop,而是有自己的*重构 loss*,那么 ACC 真的会学到那个映射。joint
cross-attention 基线(B4)被分类 loss 拉向「对分类有帮助的映射」——
事实证明,*这与恢复另一侧表征不是同一回事*。这正是 SPLIT-9 negative
result 的*原因*所在。

第二个发现:**Hebbian 在本 ML 实现中并不合适**。在 V3 中,W_hebbian
累积一种任意的外积统计,W_learned 把容量花在*抵消那个统计*,而不是
学习真实映射(η/W_max 扫描确认这是*结构性*限制,不是超参问题)。
V3 在 cosine 上比 V2 差 0.51。神经科学上忠实的直观在 ML 中*起反作用* ——
**人工胼胝体不需要 Hebbian;解耦的重构 loss + γ 策略就足够**。

这一发现与最近某些直接引入 Hebbian 风格更新的 ML 工作(Hybrid SOM
2022、HeLa-Mem 2025、Hebbian Fast Weights in ViT 2025)所采取的方向
不同。我们的发现表明,关键是*共同训练环境 + 与分类解耦的重构 loss*
这个组合;Hebbian 可能是选项,而非基础。

---

## 局限(PoC 范围)

推广本结果时需要注意的 4 点:

1. **MNIST。** 两个 modality 是同质的(都是小 CNN,都是图像的一半)。
   真正异质的设置(例如决策模型 + LLM)中行为可能不同。
2. **task 太容易。** 仅靠左侧就能 ≥95% —— 测量 #3(因果耦合)没有
   信息量。
3. **未检验「解释」。** 本 PoC 仅验证*数值上的恢复*;SPLIT-9 的
   *自然语言忠实性*是另一个问题。
4. **仅单向。** 用户假设的 full form(双向胼胝体 —— LLM 的解释反馈到
   下次决策)未被检验。

---

## 未来工作 —— SPLIT-9 reboot

既然机制在 toy 中起作用,下一个自然步骤:

1. **在 SPLIT-9 中应用 V2 形式的 ACC** —— 在 9×9 Go-Net 与冻结 LLM
   之间共同训练一个 V2 ACC。SPLIT-9 的 negative result(95% prior、
   5% 棋局相关)是否得以解决?
2. **双向胼胝体** —— 让 LLM 的解释影响 Go-Net 的下次决策。用户假设
   的 full form。从裂脑回到正常脑。
3. **D-22**:测量 #3 有信息量的更难 toy/task。
4. **D-23**:V3' "Hebbian opt-in"(`W = tanh(α)·W_h + W_l`)—— Hebbian
   是否在某些条件下有用。

本 PoC 是这条路线上的大本营 —— 定量确认机制在 toy 形式中起作用。

---

## 复现方法

```bash
# 环境(在 RTX 3070 Ti / WSL2 Ubuntu 上验证,单 8 GB GPU)
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 测试
pytest tests/ -v   # 156 项

# Day 7 全扫描 —— 5 seed × 5 epoch × 7 variant(约 20 分钟)
python scripts/run_full_sweep.py
```

结果保存在 `runs/sweep_results.json`。控制台打印 4 个表 + paired
bootstrap p 值。

要求:Python 3.10–3.12、PyTorch 2.3+、CUDA 12.1、NumPy、SciPy。

---

## 技术栈

Python 3.10–3.12 · PyTorch 2.3+ · NumPy · SciPy · pytest · 单 8 GB
消费级 GPU。CUDA 12.1。

---

## 参考文献

* Alayrac et al., *Flamingo: a Visual Language Model for Few-Shot Learning*, NeurIPS 2022.
* Haxby et al., *Hyperalignment: Modeling shared information encoded in idiosyncratic cortical topographies*, eLife 2020.
* Lambon Ralph et al., *The Roles of Left Versus Right Anterior Temporal Lobes in Semantic Memory*, Cerebral Cortex 2018.
* Koganemaru et al., paired associative stimulation 与 interhemispheric Hebbian PAS 工作。
* Innocenti & Price, *Exuberance in the development of cortical networks*, Nature Reviews Neuroscience 2005.
* Gazzaniga, *The Bisected Brain*, Appleton-Century-Crofts, 1970.
* Gazzaniga, *The Consciousness Instinct*, Farrar Straus Giroux, 2018.
* Turpin et al., *Language Models Don't Always Say What They Think*, NeurIPS 2023.
* Atanasova et al., *Faithfulness Tests for Natural Language Explanations*, ACL 2023.
* Conmy et al., *Towards Automated Circuit Discovery for Mechanistic Interpretability*, NeurIPS 2023.
* (邻接 SPLIT-9 —— post-hoc adapter negative result)

---

## 状态

PoC 完结。v0.3-day7-scenarioA。Scenario A 强验证。可以写一篇 workshop
short paper。下一个自然的步骤是 SPLIT-9 reboot(在 9×9 围棋 + LLM 上
应用 V2 形式的 ACC)。

---

## 许可证

Apache License 2.0。参见 [LICENSE](LICENSE)。

```
Copyright 2026 namdo

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

---

*「两个网络一起长大时,把它们之间的桥也一起养大。让那座桥有自己的
活儿干。这样一侧看见的东西,另一侧也会被唤醒。」* —— 本 PoC 的结论,
一句话。

---

*这是个有意思的项目。*
