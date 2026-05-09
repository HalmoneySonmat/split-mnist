# SPLIT-MNIST — 분리뇌 MNIST + 인공 뇌량 PoC

> SPLIT-9의 negative result를 받아, **공동 학습 + 인공 뇌량(ACC)**이라는
> 다음 가설을 가장 작은 toy로 검증하는 프로젝트.

**버전**: v1.2 (Day 4c 검증 완료, D-19 fix 반영)
**상태**: PoC 인프라 검증 완료. Day 5 (B2 추가 + 베이스라인 일괄) 진입 가능.

**박제 git tag**:
- `v0.0-plan` — PLAN v1.1 시점 (코드 작성 직전)
- `v0.1-day4c` — Day 1~4c 인프라 검증 완료 시점

**v1.1 → v1.2 변경 요약**
- §4.4 V3 게이트: W_learned random init의 chicken-and-egg 회피 메커니즘 명시.
- §4.5 (1): W 초기화 박제 갱신 (W_learned = 0.01·N(0,1)).
- §16.5 D-19: *발현 확정 + 수정 적용 (2026-05-10)*. 1 epoch 검증 결과 + 재학습 결과 박제.

**v1.0 → v1.1 변경 요약** (이전)
- §3 시스템 그림: 모든 텐서 shape 명시, 5개 결정사항 박제.
- §4 ACC 변형: 7개 결정사항 박제 (W 초기화, 수식, 게이트 형태 등).
- §5 평가: 4가지 측정 절차 + 정량 임계치.
- §6 베이스라인: B2를 (a)/(b)로 분리, B4 형태 확정.
- §9 hyperparameter: β sweep 범위 + 모든 fix 값.
- §10 코드 구조: 모듈 인터페이스 시그니처 박제 → §17.
- §12 시나리오: A/B/C 정량 임계치 + 근거.
- §16 신규: Deferred Experiments (D-1~D-18).
- §17 신규: 모듈 인터페이스 시그니처 (architecture spec).

---

## 0. 한 줄 요약

> 한 MNIST 숫자를 좌/우로 갈라 두 CNN이 *반쪽씩* 본다. 두 CNN과 그
> 사이의 인공 뇌량(ACC)을 *모두 from scratch로 같이 학습*해, 한쪽만
> 봐도 반대쪽 표현을 *복원*할 수 있는지 측정한다.

---

## 1. 동기 — SPLIT-9에서 무엇이 안 됐나

SPLIT-9는 이미 학습된 Go-Net과 동결된 LLM을 *사후 어댑터*로 묶었다.
결과: 학습된 어댑터의 loss 개선의 ~95%는 도메인 prior 학습이고, 보드별
신호의 인과적 기여는 ~5%였다. 정성 검사 10/10 좌표 오답.

해석: post-hoc 어댑터는 *데이터 분포가 번역하기 쉬운 신호만* 번역한다.
보드별 내용을 강제할 수 없다. 즉 분리뇌 환자처럼 좌반구가 *그럴듯한
합리화*만 만들어내는 함정에 빠진다.

---

## 2. 본 가설 (한 줄)

> 두 신경망을 *동시에 from scratch*로 학습시키며 그 사이에 *위치 무관
> coactivation 매핑*을 배우는 학습 단위(ACC)를 두면, 한쪽 자극만으로도
> 반대쪽 표현을 복원할 수 있다.

신경과학 근거:
- **Bilateral redundancy gain** (Lambon Ralph 등): 의미 기억은 양반구
  ATL에 분산 저장됨.
- **Interhemispheric Hebbian via PAS** (Koganemaru, Mima 등): 양반구
  motor cortex에 8ms 정확 타이밍의 자극을 주면 callosal 경유 Hebbian
  plasticity가 일어남.
- **Hyperalignment** (Haxby 2011, 2020): 같은 개념이 다른 anatomical
  위치에 인코딩되어 있어도 고차원 회전으로 정렬 가능.

ML 인접 작업:
- **Hybrid SOM with Hebbian binding** (2022): 두 SOM을 따로 학습 후
  Hebbian network로 묶어 cross-modal retrieval. ACC와 가장 닮음.
- **HeLa-Mem** (arXiv 2025): LLM 메모리를 Hebbian dynamic graph로
  모델링. 본 가설의 LLM 내부 적용 버전.
- **Hebbian Fast Weights in ViT** (arXiv 2025): Transformer 안 transient
  associative memory.

본 연구의 신규 component: **두 *서로 다른 도메인* 네트워크 사이에
*학습되는 뇌량*을 두는 결합 패턴**. 부품은 다 있으나 이 정확한 조합을
시도한 사례는 미확인.

---

## 3. 시스템 — 정밀화 (v1.1)

### 3.1 시스템 그림 (모든 shape 명시)

```
        MNIST 이미지 (B, 1, 28, 28)
                  │
        ┌─────────┴─────────┐
   왼쪽 14열              오른쪽 14열
   (B, 1, 28, 14)         (B, 1, 28, 14)
        ↓                       ↓
  ┌──────────┐            ┌──────────┐
  │ 좌 CNN   │            │ 우 CNN   │   ← 동일 architecture, 독립 weight
  │ Conv1(1→16, 3×3)      │ Conv1(1→16, 3×3)
  │ ReLU + Pool 2×2       │ ReLU + Pool 2×2
  │ Conv2(16→32, 3×3)     │ Conv2(16→32, 3×3)
  │ ReLU + Pool 2×2       │ ReLU + Pool 2×2
  │ Flatten (672)         │ Flatten (672)
  │ FC(672→64) + ReLU     │ FC(672→64) + ReLU
  └──────────┘            └──────────┘
   hidden_L                 hidden_R
   (B, 64)                  (B, 64)
        │                       │
        ├───── ACC ─────────────┤
        │   W ∈ ℝ^(64×64)        │
        │   ĥ_R = W·h_L          │   ← 학습:
        │   ĥ_L = Wᵀ·h_R         │     V1 Hebbian / V2 재구성 / V3 결합
        │                       │     (분류 loss와 분리 — γ 정책)
        ↓                       ↓
        └─────────┬─────────────┘
                  ↓
            concat (B, 128)
                  ↓
         분류기: FC(128→64)→ReLU→FC(64→10)
                  ↓
            logits (B, 10)
                  ↓
            cross-entropy loss
```

### 3.2 §3 결정사항 박제

| # | 항목 | 결정 |
|---|---|---|
| (1) | MNIST 분할 | **A — 14/14 정확 반, 중복 없음** (좌 0~13열, 우 14~27열) |
| (2) | 각 CNN architecture | Conv(1→16, 3×3, pad=1)→ReLU→Pool2×2 → Conv(16→32, 3×3, pad=1)→ReLU→Pool2×2 → Flatten → FC(672→64)→ReLU. 각 ~30k params. |
| (3) | Hidden 추출 지점 | **(b) — FC 후 64-dim 단일 지점**. multi-layer는 D-1로 deferred. |
| (4) | 분류기 입력 | **γ — 학습은 raw concat (α 모드, ACC 분리), 평가 시 측정 #2에서 ACC 복원 (β 모드)** |
| (5) | ACC 방향 | **양방향, 단일 W (64×64) + Wᵀ**. 비대칭 W는 D-3로 deferred. |

학습 시 두 loss가 *동시에* 흐름:
- **분류 loss**: 좌 CNN, 우 CNN, 분류기에만 backprop. ACC 안 잡힘.
- **ACC loss** (V1/V2/V3 따라): ACC만 업데이트. 좌/우 CNN의 hidden은
  *현재 forward의 detached 값*을 사용해서 ACC 학습이 좌/우 CNN backbone을
  흔들지 않게.

---

## 4. ACC (Artificial Corpus Callosum) — 정밀화 (v1.1)

ACC 핵심 자료: 좌 unit i와 우 unit j 의 *coactivation 행렬* W ∈ ℝ^(64×64).

### 4.1 세 변형 — 핵심 비교

| 변형 | W 형태 | 학습 신호 | 학습 방식 |
|---|---|---|---|
| **V1 Hebbian only** | W_hebbian | mean-centered Hebbian | `ΔW = η·(h_L−μ_L)(h_R−μ_R)ᵀ − λW`, clip. backprop 안 씀. |
| **V2 재구성 only** | W_learned (nn.Parameter) | MSE 양방향 | AdamW backprop |
| **V3 결합** | W_hebbian + tanh(g)·W_learned | 둘 다 | Hebbian rule + backprop. g=0 초기화. |

### 4.2 V1 Hebbian 정확한 수식

```
μ_L = mean(h_L, batch)        # batch별 평균, EMA 안 씀
μ_R = mean(h_R, batch)
ΔW_hebbian = η · (h_L − μ_L)(h_R − μ_R)ᵀ − λ · W_hebbian
W_hebbian ← clip(W_hebbian + ΔW_hebbian, −W_max, +W_max)
```

- η = 0.01, λ = 0.001, W_max = 1.0
- Oja, simple product (a) 등은 D-8 deferred

### 4.3 V2 재구성 loss

```
ĥ_R = W · h_L        # h_L, h_R 모두 .detach()
ĥ_L = Wᵀ · h_R
L_recon = ‖ĥ_R − h_R‖² + ‖ĥ_L − h_L‖²   # 양방향 MSE
```

Cosine, NLI 기반은 D-9 deferred.

### 4.4 V3 결합 게이트 메커니즘

```python
W = W_hebbian + tanh(g) * W_learned
# W_hebbian: not learnable, Hebbian rule로만 update.
# W_learned: nn.Parameter, backprop으로 학습. 0.01·N(0,1) 로 random init.
# g:         nn.Parameter(torch.zeros(())), 학습 가능. 0 init.
```

학습 초기에 tanh(0)=0 → 재구성 부분 영향 0 (Flamingo "no perturbation
at start" 보존) → Hebbian만으로 시작 → 점진적으로 재구성 부분 도입.

**W_learned가 0이 아닌 작은 random인 이유** — chicken-and-egg 깨기:

```
∂L/∂W_learned = ∂L/∂W · tanh(g)        # g=0 → 0 (영원히)
∂L/∂g         = ∂L/∂W · sech²(g) · W_l  # W_l=0 이면 → 0 (영원히)
```

W_learned도 0으로 두면 *둘 다 grad가 수학적으로 정확히 0*. AdamW의
epsilon이나 numerical noise로도 못 깸 (1차 모멘트가 0). 실험적으로
Day 4c 1 epoch에서 g=0.000000 그대로, recon loss 0.05 → 15108로 폭발
확인됨.

W_learned에 작은 random을 두면 `∂L/∂g = ∂L/∂W · 1 · W_learned ≠ 0` 이
되어 g가 grad 받기 시작 → tanh(g) > 0 → W_learned도 grad 받음 →
대칭성 깨짐.

### 4.5 §4 결정사항 박제

| # | 항목 | 결정 |
|---|---|---|
| (1) | W 초기화 | **W_hebbian = 0, g = 0, W_learned = 0.01·N(0,1)** (D-19 회피, §16.5 참조) |
| (2) | Hebbian 수식 | **(b) mean-centered, batch별 평균** |
| (3) | 재구성 loss | **MSE 양방향 합** |
| (4) | V3 게이트 | **`W = W_hebbian + tanh(g)·W_learned`, g=0 초기화** |
| (5) | hidden detach | **양쪽 모두 `.detach()` (PoC). 공동 진화는 D-10.** |
| (6) | optimizer | **분류기/CNN과 같은 AdamW, lr=1e-3** |
| (7) | V1 분류기 입력 | **모든 변형에서 raw concat (γ 정책 일관). V1도 분류 정확도 정상.** |

---

## 5. 평가 — 정밀화 (v1.1)

### 5.1 측정 #1 — Task accuracy

표준 분류 정확도. test 10000장. 5 seed 평균 ± std. `92.3 ± 0.4 %` 형식.

### 5.2 측정 #2 — Cross-activation faithfulness ★ 핵심

**Ablation 방법**: **Mean ablation**. 우 hidden을 train 평균 μ_R로 치환.
Resampling은 D-13 deferred.

**비교 대상**: 두 가지 *모두* 측정.
- (i) **표현 수준 cosine**: cos(W·h_L, h_R 진짜)
- (ii) **행동 수준 분류 정확도 회복**: classifier(cat[h_L, ĥ_R]) accuracy

**양방향**: 좌 ablation/우 복원, 우 ablation/좌 복원 모두 측정. 결과 표:
```
                  cosine    classifier acc
좌 가리고 좌 복원:   0.XX        XX.X %
우 가리고 우 복원:   0.XX        XX.X %
```

### 5.3 측정 #3 — Causal coupling

**Noise**: 가우시안. h_L ← h_L + ε·N(0,1).
**Sweep**: ε ∈ {0, 0.1, 0.25, 0.5, 1.0, 2.0}.
**보고**: ε vs accuracy 곡선 (figure) + ε=0.5 단일 숫자 (표).

### 5.4 측정 #4 — Position invariance

**비교 방식**: 5 seed의 학습 후 W들.
- (i) Heatmap 5장 시각 비교 (figure)
- (ii) Procrustes alignment 후 상관계수 r (정량)

RSA 분석은 D-14 deferred.

### 5.5 정량 임계치 사전 등록

| 측정 | 임계치 (사전 등록) | 근거 |
|---|---|---|
| #2 cosine | ≥ 0.7 (강), 0.3~0.7 (약), <0.3 (실패) | hyperalignment 표준 |
| #2 V3 vs B4 cosine 차이 | ≥ 0.15, paired bootstrap p < 0.05 | SBERT 의미 거리 임계 0.2보다 보수 |
| #3 V3 vs B3 ε=0.5 acc 하락 차이 | ≥ 5%p | SPLIT-9 IAS 보고 패턴 |
| #4 Procrustes r | ≤ 0.3 (위치 무관 OK), > 0.7 (수렴 발견) | 일반 약/강 상관 임계 |

### 5.6 통계 처리 (모든 측정 공통)

| 항목 | 결정 |
|---|---|
| Seed | 5 (42~46) |
| 평균 보고 | mean ± std |
| 본 모델 vs 베이스라인 유의성 | paired bootstrap, n=10000 |
| 다중 비교 보정 | Holm-Bonferroni (V1/V2/V3 × B1~B4) |
| Cosine CI | 95% bootstrap |

---

## 6. 베이스라인 — 정밀화 (v1.1)

| 베이스라인 | 두 CNN | hidden 결합 | 학습 신호 | 측정 #2 적용? |
|---|---|---|---|---|
| **B1 Single CNN** | 1개 (28×28 통째) | — | 분류 loss | 불가 |
| **B2(a) Independent-avg** | 2개 | 각자 분류기 → logit 평균 | 분류 loss | 불가 |
| **B2(b) Left-only** | 2개 (B2(a) 유도) | 좌 CNN만 사용 | 분류 loss | 불가 |
| **B3 Concat** | 2개 | concat → 분류기 | 분류 loss | 불가 |
| **B4 Cross-attn** | 2개 | **1-head 양방향 cross-attn** | 분류 loss | **가능** ★ |
| **본 모델 V1** | 2개 | concat (학습), ACC (평가 #2) | 분류 + Hebbian | 가능 |
| **본 모델 V2** | 2개 | concat, ACC | 분류 + 재구성 | 가능 |
| **본 모델 V3** | 2개 | concat, ACC | 분류 + Hebbian + 재구성 | 가능 |

**B1 형태**: 28×28 통째 입력, Conv(1→16)→Pool→Conv(16→32)→Pool→FC(...)→FC(64→10).
*상한 reference*. parameter fair는 안 따짐.

**B4 정확한 형태**: 1-head, dim 64, *양방향* cross-attention.
- `h_L' = h_L + Attn(Q=h_L, K=V=h_R)`
- `h_R' = h_R + Attn(Q=h_R, K=V=h_L)`
- W_Q, W_K, W_V, W_O 각 64×64 → ~16k params (ACC W 4k 대비 4배)
- 분류 loss로 backprop, ACC와 다르게 *분류기와 묶여서* 학습
- Multi-head는 D-17 deferred

**핵심 비교 두 개**:
1. 본 모델 V3 vs B3 Concat — "ACC 추가가 단순 결합보다 나은가?"
2. 본 모델 V3 vs B4 Cross-attn — "ACC의 분리 학습 + Hebbian이 단순
   cross-attn보다 측정 #2~#4에서 우월한가?" ★ 결정적

---

## 7. 데이터셋과 분할

- MNIST 60k train / 10k test (표준).
- Train 50k / val 10k 으로 train 재분할 (seed 고정).
- 좌 14×28 / 우 14×28 단순 split. 가운데 1열 중복 D-2 deferred.
- Normalization: mean=0.1307, std=0.3081 (MNIST 표준)

---

## 8. 모델 사이즈 (8GB GPU에서 여유 있음)

- 좌/우 CNN 각: ~30k params
- ACC W: 4k params (V1/V2)
- ACC V3: ~8k params (W_hebbian + W_learned + g)
- B4 Cross-attn: ~16k params
- 분류기: ~10k params
- 전체: 75~85k params. 사실상 무시 크기.

---

## 9. 학습 hyperparameter — 정밀화 (v1.1)

### 9.1 Sweep (1개만)

| Hyperparam | Sweep 범위 | 적용 변형 |
|---|---|---|
| **β (재구성 loss 가중치)** | {0.1, 0.5, 1.0, 2.0} | V3만. V1은 β 무관, V2는 β=1.0 fix. |

### 9.2 Fix 값

| Hyperparam | 값 |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 128 |
| Epochs | 20 (early stopping val acc, patience=3) |
| Hebbian η | 0.01 |
| Hebbian decay λ | 0.001 |
| W_max (clip) | 1.0 |
| 게이트 g 초기값 | 0.0 (V3) |
| Hidden dim | 64 |
| Seeds | 5 (42, 43, 44, 45, 46) |
| Augmentation | Normalize만 |

### 9.3 Loss 결합

```python
L_total = L_classification + β * L_reconstruction
L_total.backward()
optimizer.step()
acc.hebbian_update(h_L.detach(), h_R.detach())  # V1, V3에만, no grad
```

### 9.4 학습 횟수 추산

- B1, B2/B3, B4, V1, V2 (β=1.0) × 5 seed = 5 × 5 = 25
- V3 × 5 seed × 4 β = 20
- **합계 45 학습 × ~30분 ≈ 22.5시간 GPU** (RTX 3070 Ti)

---

## 10. 코드 구조

```
split_mnist/
├── PLAN.md                  ← 본 문서 (v1.1)
├── README.md                ← Phase 0 종료 후 결과 요약
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── src/split_mnist/
│   ├── __init__.py
│   ├── data.py              ← MNIST split 로더
│   ├── networks.py          ← HalfCNN, Classifier, SingleCNN
│   ├── acc.py               ← ACCBase, ACCv1Hebbian, ACCv2Recon, ACCv3Combined, CrossAttnAdapter
│   ├── losses.py            ← classification_loss
│   └── train.py             ← TrainConfig, build_model, train_one_run
├── scripts/
│   ├── train.py             ← 본 모델/베이스라인 진입점 (CLI)
│   ├── train_baselines.py   ← B1~B4 일괄 학습
│   ├── evaluate.py          ← 측정 #1~#4 일괄 실행
│   └── make_figures.py      ← Heatmap, ε 곡선 figure 생성
├── tests/
│   ├── test_data.py
│   ├── test_networks.py
│   ├── test_acc_v1_hebbian.py
│   ├── test_acc_v2_recon.py
│   ├── test_acc_v3_combined.py
│   ├── test_crossattn.py
│   └── test_train_smoke.py
└── runs/
    ├── checkpoints/
    └── results/
```

분량 추산: 1500~2000줄 (v1.0 추산보다 약간 늘어남, ablation 변형 늘어서).

---

## 11. 진행 일정

| Day | 작업 |
|---|---|
| 1 | data.py, networks.py + 단위 테스트 |
| 2 | acc.py V2 (재구성, 가장 단순) + 단위 테스트 |
| 3 | acc.py V1 (Hebbian) + V3 (결합) + 단위 테스트 |
| 4 | acc.py CrossAttnAdapter (B4) + train.py + 1 epoch smoke 학습 |
| 5 | scripts/train.py + 베이스라인 B1~B4 |
| 6 | evaluate.py — 측정 #1~#4 |
| 7 | 본 모델 V1/V2/V3 + 베이스라인 5 seed씩 학습 |
| 8 | β sweep (V3 × 4 β × 5 seed) + 결과 종합 + figures |

총 8일 (PLAN v1.0 7일에서 1일 늘어남, B4 cross-attn + β sweep 추가).

---

## 12. 결과 시나리오 — 정밀화 (v1.1, 사전 등록)

### Scenario A — 가설 강한 검증 (모두 만족)

| 측정 | 임계치 |
|---|---|
| #1 Task accuracy | V3 ≥ B1 − 5%p |
| #2 Cross-activation cosine | V3 ≥ 0.7 |
| #2 V3 vs B4 cosine 차이 | ≥ 0.15, paired bootstrap p < 0.05 |
| #2 V3 vs B4 classifier acc 차이 | ≥ 5%p |
| #3 Causal coupling | V3 ε=0.5 acc 하락 − B3 acc 하락 ≥ 5%p |
| #4 Position invariance | Procrustes r ≤ 0.3 across 5 seeds |

→ **다음 단계**: SPLIT-9 reboot (9×9 바둑 + LLM에 ACC 적용). 워크숍 short paper.

### Scenario B — 부분 검증 (어느 하나라도 해당)

| 패턴 | 의미 |
|---|---|
| #2 V3 cosine ≥ 0.5 BUT V3 vs B4 차이 < 0.15 | "ACC 동작은 하나 Hebbian이 cross-attn 대비 우월하지 않음" |
| #4 Procrustes r ∈ [0.3, 0.7] | "어느 정도 위치 수렴 — 완전 위치 무관 아님" |
| #2 만족이지만 #3 만족 안 함 | "표현 복원은 되나 인과 결합 약함" |
| V1/V2/V3 중 한두 변형만 통과 | 부분 검증 |

→ **다음 단계**: short paper 가능. SPLIT-9 reboot 전 추가 ablation (D-3, D-8, D-10, D-13 등).

### Scenario C — 가설 기각 (어느 하나라도 해당)

| 패턴 | 의미 |
|---|---|
| #2 V3 cosine < 0.3 | "복원 실패" |
| #1 V3 < B3 (p < 0.05) | "ACC 추가가 분류 성능 저하" |
| #2 V3 cosine ≤ B4 cosine (p < 0.05) | "Hebbian이 cross-attn보다 못함" |
| #3 V3 곡선이 B3와 통계적으로 동일 | "인과 결합 없음" |

→ **다음 단계**: 두 번째 negative result. SPLIT-9 패턴으로 negative paper 작성. 다음 가설로 회귀 (예: ACC sparse top-k, 공동 진화 모드 D-10).

### 12.4 임계치 근거

| 임계치 | 근거 |
|---|---|
| V3 cosine ≥ 0.7 | hyperalignment 논문에서 "good alignment"의 일반 기준 |
| V3 vs B4 차이 ≥ 0.15 | SBERT 의미 거리에서 "의미 다름"의 통상 임계 0.2보다 보수 |
| #1 V3 ≥ B1 − 5%p | 분리뇌에서 단편 정보로 5%p 손실은 받아들일 수준 |
| #3 ε=0.5 5%p 차이 | SPLIT-9 IAS 보고 패턴 |
| #4 r ≤ 0.3 / ≥ 0.7 | 약/강 상관의 일반 임계 |
| paired bootstrap p < 0.05 | 표준 |

세 시나리오 모두 *발표 가능한 결과*. 즉 무엇이 나와도 살아남는다.

---

## 13. 무엇을 하지 않을 것인가 (PoC 범위 밖)

- **LLM은 안 씀**. 분류 head는 단순 MLP. 자연어 설명은 PoC 범위 밖.
- **MCTS, self-play 안 씀**. supervised classification만.
- **Pretrained model 안 씀**. 모두 from scratch (가설의 본형 검증).
- **인간 평가 안 씀**. 측정 #1~#4 모두 자동.

---

## 14. 다음 단계 (PoC 후)

**Scenario A 시**:
1. 본 PoC 결과를 short paper (워크숍) 로 정리.
2. SPLIT-9 reboot — 9×9 바둑 + LLM에 ACC 적용. 결정자/해석자가 진짜로
   이질적인 환경에서도 동작하는지 검증.
3. 그 다음 양방향 — LLM 설명이 Go-Net 다음 결정에 영향. 진짜 뇌량.

**Scenario B/C 시**: 결과에 따라 deferred 항목 중 일부 시도.

---

## 15. 사전 등록 (Pre-registered)

본 문서의 측정 절차 (§5), 베이스라인 (§6), hyperparameter (§9),
시나리오 임계치 (§12) 는 학습 시작 전에 박제됨. 학습 결과 본 후
임계치 변경 금지.

git tag `v0.0-plan` 으로 박제 예정 (사용자 환경에서 직접 실행).

만약 학습 도중 *명백히 잘못된 임계치*가 발견되면 (예: 모든 V3가 cosine
0.85인데 B4도 0.83이라 임계치 0.15가 너무 빡빡) → 임계치 조정은
가능하나 본 문서에 *왜 조정했는지* 명시하고 별도 섹션 "Post-hoc
adjustments"에 기록.

---

## 16. Deferred Experiments (D-1 ~ D-18)

PoC에서는 안 하지만 *후속 실험에서 가치 있는* 항목들. 본 PoC 결과
(Scenario A/B/C) 본 후 어느 것을 시도할지 결정. 영원히 안 하는 게
아니라 *지금 안 한다*.

### 16.1 시스템 구조 (§3에서 미룸)

- **D-1 Multi-layer ACC** — Conv2 출력 + FC 출력 둘 다에서 hidden 추출.
  여러 층의 시냅스 짝짓기. 사용자 가설의 *깊은 형태*.
- **D-2 비대칭 분할** — 13/15 또는 가운데 1열 중복.
- **D-3 비대칭 W** — ACC_LR과 ACC_RL을 *서로 다른* 행렬로. 표현력 ↑,
  본 가설("뇌량은 *대응 시냅스 쌍*을 하나의 매핑으로")에서 멀어짐.
- **D-4 더 큰 hidden dim** — 128, 256, 512.
- **D-5 Multi-layer CNN 깊이 확장** — ResNet 잔차.
- **D-6 다른 도메인** — CIFAR, 음성-이미지, 다국어 텍스트 등.
- **D-7 좌/우 CNN architecture 비대칭** — 좌 = 작은 dense, 우 = 큰 sparse
  등. 인간 뇌 좌/우 비대칭 모방.

### 16.2 ACC 변형 (§4에서 미룸)

- **D-8 Oja 정규화 Hebbian rule** — 자동 norm 제어 PCA 학습. (b)와 비교.
- **D-9 Cosine 또는 NLI/SBERT 기반 재구성 loss** — MSE 대안.
- **D-10 ACC ↔ CNN 공동 진화** — 둘 다 grad 흐름 (detach 안 함).
  표현이 *공동 진화*하는 더 강한 형태.
- **D-11 ACC 별도 optimizer / lr scheduler** — 미세 제어.
- **D-12 Multi-step Hebbian** — 한 batch 안 여러 update.
- **D-16 ACC W에 sparsity 제약** — top-k 또는 L1. "1:1 짝꿍이 더 나은가?"

### 16.3 평가 (§5에서 미룸)

- **D-13 Resampling ablation** — mean ablation 대신 다른 sample의 hidden.
- **D-14 RSA (representational similarity analysis)** — W matrix 간
  similarity matrix 비교.
- **D-15 Adversarial perturbation 기반 #3 측정** — random noise 대신
  adversarial.

### 16.4 베이스라인 (§6에서 미룸)

- **D-17 B4 multi-head** (8-head, dim 8 each) — capacity 더 큰 비교군.
- **D-18 B5 — Frozen pretrained CNN 베이스라인** — SPLIT-9 패턴 직접 재현.
  PoC 후 SPLIT-9 reboot 단계에서 자연스럽게 추가.

### 16.5 발견된 이슈 (학습 시점 검증 완료)

- **D-19 V3 chicken-and-egg 대칭성** — *발현 확정 + 수정 적용 (2026-05-10)*

  PLAN §4.5 (1) 원안에서 W_h, W_l, g 모두 0 초기화했음. 이 상태에선
  `g=0 → tanh(0)=0 → W_l grad path 무력`, 동시에
  `W_l=0 → g grad = sech²(g)·(∂L/∂W·W_l) = 0`. 둘 다 grad가 *수학적으로
  정확히 0*. AdamW의 epsilon으로도 못 깸 (1차 모멘트가 0).

  **Day 4c 검증 결과 (V3 1 epoch on MNIST)**:
  - val_acc 95.41% (B4와 동일) — 분류 path는 정상 학습
  - **final g = 0.000000** (영원히 0)
  - **loss_recon: 0.05 → 15108** (폭발)
    - 이유: W_hebbian이 *분류와 무관하게* Hebbian으로만 자라며 임의의
      매핑에 수렴 → ĥ_R 폭발 → MSE 폭발. W_learned가 죽어 있어 보정
      불가능.

  **수정 적용**: `W_learned = 0.01 · N(0, 1)` 로 random init.
  - `tanh(0) · W_learned = 0` 이라 학습 시작 시점 영향 0 (Flamingo 게이트
    패턴 유지)
  - 그러나 `∂L/∂g = ∂L/∂W · sech²(0) · W_learned ≠ 0` → g가 grad 받음
    → 대칭성 깨짐
  - 코드 변경: `acc.py` ACCv3Combined.__init__의 W_learned 초기화 한 줄.
    PLAN §4.4 / §4.5 (1) 갱신.

  **사후 검증 결과 (Day 4c 재학습, 2026-05-10)** — **fix 효과 확인**:

  | 지표 | V3 broken (이전) | V3 fixed (재학습) |
  |---|---|---|
  | step 0 g | 0.0000 | -0.0010 (random init 영향) |
  | step 100 g | 0.0000 | -0.0924 |
  | step 200 g | 0.0000 | -0.2718 |
  | step 350 g | 0.0000 | -0.4720 |
  | final g | 0.000000 | **-0.510018** (단조 학습) |
  | step 100 recon | 14,356 | 21,484 (정점) |
  | step 350 recon | 15,651 (계속 폭발) | **6,976 (감소 시작)** |
  | final recon | 15,108 | 5,800 |
  | val_acc | 95.41% | 96.44% |
  | test_acc | 95.97% | 96.81% |

  **해석**:
  - g가 *단조* 학습 — AdamW가 진짜로 게이트를 학습 중 (random 진동 X).
  - g 음수 부호 — W_learned가 W_hebbian의 *임의 매핑을 빼주는* 방향으로
    학습. effective W의 norm이 작아져 recon loss 감소.
  - recon loss 정점 후 감소 — 처음에는 W_hebbian이 임의 매핑으로 자라며
    recon이 폭발하지만, W_learned의 backprop이 점진적으로 *진짜 매핑에
    가깝게* W를 보정. PLAN §4.4 narrative *"Hebbian이 초기 모양 형성,
    재구성으로 미세 조정"* 확인.
  - 분류 성능 영향 없음 (또는 살짝 향상). γ 정책의 detach 덕분에 ACC
    학습이 backbone을 망치지 않음 — 이것도 §3.2 (4) 결정의 정확한 작동.

  **결론**: D-19 fix 효과적. PoC 인프라 통합 검증 완료.

---

## 17. 모듈 인터페이스 시그니처 (architecture spec)

코드 작성 시 *바꾸지 않을* 시그니처. SPLIT-9의 docs/architecture.md
패턴을 본 PLAN.md에 직접 통합.

### 17.1 `data.py`

```python
import torch
from torch.utils.data import Dataset, DataLoader

class SplitMNIST(Dataset):
    """
    Split each MNIST image into left half (cols 0-13) and right half (cols 14-27).
    Returns (x_left, x_right, y).
    """
    def __init__(
        self,
        root: str,
        train: bool = True,
        normalize: bool = True,
    ): ...

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        # Returns: x_left (1, 28, 14), x_right (1, 28, 14), y (int 0..9)
        ...

def make_loaders(
    root: str = "./data",
    batch_size: int = 128,
    val_size: int = 10_000,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader). Train split 50k/10k."""
    ...
```

### 17.2 `networks.py`

```python
import torch.nn as nn
from torch import Tensor

class HalfCNN(nn.Module):
    """One half of the bilateral system. Same arch for left and right; weights independent."""
    def __init__(self, hidden_dim: int = 64): ...
    def forward(self, x: Tensor) -> Tensor:
        # x: (B, 1, 28, 14) -> hidden (B, 64)
        ...

class Classifier(nn.Module):
    """Takes concat[hidden_L, hidden_R] -> 10 logits."""
    def __init__(self, hidden_dim: int = 64, n_classes: int = 10): ...
    def forward(self, hidden_L: Tensor, hidden_R: Tensor) -> Tensor:
        # Returns logits (B, 10)
        ...

class SingleCNN(nn.Module):
    """B1 baseline. Takes full 28x28 image."""
    def __init__(self, hidden_dim: int = 64, n_classes: int = 10): ...
    def forward(self, x: Tensor) -> Tensor:
        # x: (B, 1, 28, 28) -> logits (B, 10)
        ...
```

### 17.3 `acc.py`

```python
import torch
import torch.nn as nn
from torch import Tensor

class ACCBase(nn.Module):
    """Abstract base for all ACC variants."""
    def forward_LR(self, hidden_L: Tensor) -> Tensor:
        """Predict hidden_R from hidden_L."""
        ...
    def forward_RL(self, hidden_R: Tensor) -> Tensor:
        """Predict hidden_L from hidden_R."""
        ...
    def reconstruction_loss(self, hidden_L: Tensor, hidden_R: Tensor) -> Tensor:
        """Bidirectional MSE. Returns scalar tensor."""
        ...
    def hebbian_update(self, hidden_L: Tensor, hidden_R: Tensor) -> None:
        """In-place update of W_hebbian. No-op for variants without Hebbian."""
        ...

class ACCv1Hebbian(ACCBase):
    """Pure Hebbian: W = W_hebbian only, no backprop."""
    def __init__(
        self,
        hidden_dim: int = 64,
        eta: float = 0.01,
        decay: float = 0.001,
        w_max: float = 1.0,
    ): ...

class ACCv2Recon(ACCBase):
    """Pure reconstruction: W = W_learned (nn.Parameter), backprop only."""
    def __init__(self, hidden_dim: int = 64): ...

class ACCv3Combined(ACCBase):
    """W = W_hebbian + tanh(g) * W_learned. Both update paths active."""
    def __init__(
        self,
        hidden_dim: int = 64,
        eta: float = 0.01,
        decay: float = 0.001,
        w_max: float = 1.0,
    ): ...

class CrossAttnAdapter(nn.Module):
    """B4 baseline. 1-head bidirectional cross-attention. Trained with classification loss only."""
    def __init__(self, hidden_dim: int = 64): ...
    def forward(
        self, hidden_L: Tensor, hidden_R: Tensor
    ) -> tuple[Tensor, Tensor]:
        # Returns: hidden_L', hidden_R' (each (B, 64))
        ...
```

### 17.4 `losses.py`

```python
def classification_loss(logits: Tensor, y: Tensor) -> Tensor:
    """Standard cross-entropy."""
    ...

# (재구성 loss와 Hebbian update는 ACC 클래스의 메서드로.)
```

### 17.5 `train.py`

```python
from dataclasses import dataclass

@dataclass
class TrainConfig:
    seed: int = 42
    n_epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    beta_recon: float = 1.0
    early_stop_patience: int = 3
    variant: str = "V3"   # "V1" | "V2" | "V3" | "B1" | "B2" | "B3" | "B4"
    out_dir: str = "runs/"

def build_model(cfg: TrainConfig) -> dict:
    """
    Returns dict with keys depending on variant:
      base ("V1"~"V3", "B4"): {"left", "right", "classifier", "acc"}
      "B1": {"single_cnn"}
      "B2"/"B3": {"left", "right", "classifier"}
    """
    ...

def train_one_run(cfg: TrainConfig) -> dict:
    """
    Full training loop. Returns metrics dict + saves checkpoints.
    Metrics: {"val_acc_curve": [...], "best_val_acc": float, "best_test_acc": float}
    """
    ...
```

### 17.6 `evaluate.py` (in scripts/)

```python
def measure_task_accuracy(model_dict, test_loader, device) -> float: ...

def measure_cross_activation(
    model_dict, test_loader, device,
    direction: str = "right_ablation",  # or "left_ablation"
) -> dict:
    """Returns: {"cosine": float, "classifier_acc_recovered": float}"""
    ...

def measure_causal_coupling(
    model_dict, test_loader, device,
    epsilons: list[float] = [0, 0.1, 0.25, 0.5, 1.0, 2.0],
) -> dict:
    """Returns: {"epsilon_curve": list[float], "ias_at_0p5": float}"""
    ...

def measure_position_invariance(
    W_list: list[torch.Tensor],   # 5 seeds
) -> dict:
    """Returns: {"procrustes_r": float, "heatmap_paths": list[str]}"""
    ...
```

### 17.7 의존 방향

```
data ──→ train
           │
networks ──┤
           │
acc ──────┤
           │
losses ────┘

evaluate uses everything (read-only)
```

원칙:
- `data`, `networks`, `acc`, `losses`는 *서로 import 안 함*. 조립은 `train`이 한다.
- `evaluate`는 모든 모듈 읽기 전용 사용. 학습 객체 변경 금지.
- 모든 텐서는 caller가 device 관리. 모듈 내부에서 `.cuda()` 호출 안 함.

---

*v1.1 작성일: 2026-05-09. PLAN v1.0 + §3~§12 정밀화 통합. 코드 작성 직전 상태.*
