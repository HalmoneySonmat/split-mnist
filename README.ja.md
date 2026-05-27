# SPLIT-MNIST

[🇰🇷 한국어](README.md) · [🇬🇧 English](README.en.md) · [🇨🇳 中文](README.zh.md) · 🇯🇵 日本語

*MNIST を左右に分割し、二つの小さな CNN とその間の学習可能な人工脳梁を同時に訓練したとき、片側の刺激だけで反対側の表現を復元できるかを検証した PoC。*

前プロジェクト SPLIT-9(事後アダプタ)が忠実性の構造的天井に当たり陰性で終わった後、その天井を破る仮説 ── 二つの網を from scratch で *同時* 学習し、間に *学習される人工脳梁(ACC)* を置く ── を最小の toy(左右に分けた MNIST)で検証した。
仮説本形(V2、分離された再構成 loss)は片側 hidden を ablation しても反対側を cosine 0.81 / 分類 0.83 まで復元 ── 事後アダプタ baseline(B4)に対し cosine +0.16、acc_recon +0.11(いずれも p<0.0001)。事前登録の閾値 6 個中 5 個通過。
一方、神経科学的に自然な Hebbian(V3)は ML 実装では *ノイズ* として働いた(V2 比 cosine −0.51)── それ自体が発見。メカニズムが toy で動くことを定量的に示した **陽性** 結果。

**見出し:** V2 cosine 0.81 / acc_recon 0.83 · vs B4 +0.16 / +0.11(p<0.0001) · 位置不変 r 0.46 vs random 0.75 · 事前登録 5/6 通過(Scenario A 強い検証)。

**[詳細を見る → 手法・測定 4 種・表(韓/英)](https://halmoneysonmat.github.io/split-mnist/)**

---

### 研究ラインアップ

SPLIT-9 の陰性がこの PoC を動機づけ、この陽性が次の一歩へつながる ── 二つの網を最初から *一緒に* 育てる co-developed twins。

[SPLIT-9](https://github.com/HalmoneySonmat/split-9) → **SPLIT-MNIST** → [SPLIT-MAZE](https://github.com/HalmoneySonmat/split-maze)

再現方法・スタック・参考文献は詳細ページに。テスト: `pytest tests/ -v`(~156 個)。

---

このプロジェクトの作成に Claude を使った。
