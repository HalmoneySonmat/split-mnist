# SPLIT-MNIST

[🇰🇷 한국어](README.md) · [🇬🇧 English](README.en.md) · 🇨🇳 中文 · [🇯🇵 日本語](README.ja.md)

*把 MNIST 切成左右两半,让两个小 CNN 和它们之间可训练的人工胼胝体同时从零训练,验证仅凭一侧的刺激能否恢复另一侧的表征的 PoC。*

前一个项目 SPLIT-9(事后适配器)因忠实性的结构性天花板以阴性收尾;本 PoC 在最小的 toy(左右切分的 MNIST)上验证打破该天花板的假设 —— 把两个网络 from scratch *同时* 训练,并在它们之间放一个 *可训练的人工胼胝体(ACC)*。
假设本形(V2,带分离的重建 loss)在对一侧 hidden 做 ablation 后,仍能把另一侧恢复到 cosine 0.81 / 分类 0.83 —— 相对事后适配器 baseline(B4):cosine +0.16,acc_recon +0.11(均 p<0.0001)。预注册 6 个阈值过 5 个。
而神经科学上自然的 Hebbian(V3)在 ML 实现中却成了 *噪声*(相对 V2 cosine −0.51)—— 这本身就是一项发现。这是定量证明该机制在 toy 上奏效的 **阳性** 结果。

**头条:** V2 cosine 0.81 / acc_recon 0.83 · vs B4 +0.16 / +0.11(p<0.0001) · 位置无关 r 0.46 vs random 0.75 · 预注册 5/6 通过(Scenario A 强验证)。

**[查看详情 → 方法 · 四项测量 · 表格(韩/英)](https://halmoneysonmat.github.io/split-mnist/)**

---

### 研究脉络

SPLIT-9 的阴性推动了这个 PoC,而这个阳性又引向下一步 —— 从一开始就把两个网络 *一起* 养大(co-developed twins)。

[SPLIT-9](https://github.com/HalmoneySonmat/split-9) → **SPLIT-MNIST** → [SPLIT-MAZE](https://github.com/HalmoneySonmat/split-maze)

复现方法、技术栈与参考文献见详情页。测试: `pytest tests/ -v`(~156 个)。

---

本项目的搭建使用了 Claude。
