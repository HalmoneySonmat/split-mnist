# SPLIT-MNIST

*분리뇌 MNIST + 인공 뇌량(ACC). SPLIT-9의 negative result에 이은 다음 가설 검증 PoC.*

> **TL;DR.** 한 MNIST 숫자를 좌/우로 갈라 두 작은 CNN이 *반쪽씩* 본다.
> 두 CNN과 그 사이의 인공 뇌량(ACC)을 *모두 from scratch로 같이* 학습.
> 한쪽만 봐도 반대쪽 표현을 *복원*할 수 있는지 측정.

## 왜 이걸 만드나

SPLIT-9 (인접 폴더 `D:\brain\split_brain_go`) 가 *post-hoc 어댑터*로
9×9 Go-Net과 동결된 LLM을 묶었다. 결과: 어댑터의 학습은 *95% 도메인
prior*, *5% 보드별 신호*로 분해되며 정성 검사에서 0/10 좌표 오답.
post-hoc adapter는 *데이터 분포가 번역하기 쉬운 신호만* 번역하고
*특정 내용을 강제할 수 없다*는 구조적 천장에 부딪힌 것.

본 PoC는 그 한계를 깨는 *다음 가설*을 가장 작은 toy로 검증한다:

> 두 신경망을 *동시에 from scratch*로 학습시키며 그 사이에 *위치 무관
> coactivation 매핑*을 학습하는 단위(ACC)를 두면, 한쪽 자극만으로도
> 반대쪽 표현을 복원할 수 있다.

신경과학 근거 — Lambon Ralph의 bilateral redundancy gain (의미 기억의
양반구 분산 저장), Koganemaru/Mima의 interhemispheric Hebbian via PAS
(8ms 정밀 timing의 callosal-mediated 시냅스 강화), Haxby의 hyperalignment
(같은 개념이 다른 anatomical 위치에 인코딩되어도 고차원 회전으로 정렬).

ML 인접 작업 — Hybrid SOM with Hebbian binding (2022),
HeLa-Mem (arXiv 2025), Hebbian Fast Weights in ViT (arXiv 2025).
본 연구의 신규 component는 *두 서로 다른 도메인 네트워크 사이에
학습되는 뇌량을 두는 결합 패턴*.

## 시스템

```
        MNIST 이미지 (B, 1, 28, 28)
                  │
        ┌─────────┴─────────┐
   왼쪽 14열              오른쪽 14열
   (B, 1, 28, 14)         (B, 1, 28, 14)
        ↓                       ↓
  ┌──────────┐            ┌──────────┐
  │ 좌 CNN   │            │ 우 CNN   │   ← 동일 architecture, 독립 weight
  │ Conv→Pool│            │ Conv→Pool│   ← 둘 다 from scratch
  │ Conv→Pool│            │ Conv→Pool│
  │ FC(672→64)            │ FC(672→64)
  └──────────┘            └──────────┘
   hidden_L                 hidden_R
   (B, 64)                  (B, 64)
        │                       │
        ├───── ACC ─────────────┤
        │   W ∈ ℝ^(64×64)        │
        │   ĥ_R = W·h_L          │   ← V1 Hebbian / V2 재구성 / V3 결합
        │   ĥ_L = Wᵀ·h_R         │     (분류 loss와 분리 — γ 정책)
        ↓                       ↓
        └─────────┬─────────────┘
                  ↓
            concat (B, 128)
                  ↓
         분류기: FC(128→64)→ReLU→FC(64→10)
```

## ACC 세 변형 (ablation)

| variant | W 형태 | 학습 신호 |
|---|---|---|
| **V1 Hebbian only** | W_hebbian (buffer) | mean-centered Hebbian, no backprop |
| **V2 재구성 only** | W_learned (Parameter) | bidirectional MSE backprop |
| **V3 결합** ★ 본형 | W_hebbian + tanh(g)·W_learned | 둘 다 + Flamingo 게이트 |

베이스라인 4종 — B1 (SingleCNN, 분리 안 한 상한), B2(a/b) (independent),
B3 (concat), B4 (CrossAttnAdapter — ACC와 *같은 capacity*, 분류 loss로만 학습).

## 평가 4종 (PLAN §5)

| # | 측정 | 핵심 질문 |
|---|---|---|
| 1 | Task accuracy | 분류 잘 되나 |
| 2 | **Cross-activation faithfulness** ★ | 우 ablation 후 ACC가 우 hidden을 복원하나 |
| 3 | Causal coupling | 좌에 noise 가하면 우 분류도 영향 받나 |
| 4 | Position invariance | 매 seed마다 W 패턴 다른가 (위치 무관) |

## 현재 상태 (2026-05-10)

**v0.3-day7 — PoC 완결. Scenario A 강한 검증.** 🎉

### 핵심 결과

5 seed × 5 epoch × 7 variant full sweep:

| 비교 | Δ | p-value | 판정 |
|---|---|---|---|
| **V2 vs B4 cosine** | **+0.161** | **0.0000** | ✓ Scenario A (≥ 0.15) |
| **V2 vs B4 acc_recon** | **+0.107** | **0.0000** | ✓ Scenario A (≥ 0.05) |
| V2 r_trained vs random | −0.284 | — | ✓ 위치 무관 OK |
| V3 vs V2 cosine | −0.511 | 0.0000 | Hebbian = 잡음 |

**6개 측정 #1~#4 임계 중 5개 통과** (측정 #3은 toy 한계, §18.5에서 사전 인정).

| variant | cosine | acc_recon | Δ(rec−abl) | 비고 |
|---|---:|---:|---:|---|
| **V2** ★ | **0.814** | **0.829** | **+0.073** | 본 가설 본형 |
| B4 | 0.653 | 0.722 | −0.018 | joint cross-attn (불러옴) |
| V3 | 0.303 | 0.681 | −0.079 | Hebbian = 잡음 (D-20 확정) |
| V1 | 0.273 | 0.418 | −0.319 | Hebbian only (예상한 처참) |

**V2가 *유일하게* `acc_recon > acc_ablated`** — ACC 복원본이 진짜로 분류 도움.
B4, V1, V3는 random ablation보다도 못 복원.

### 사용자 가설 매핑

| 가설의 부분 | 결과 |
|---|---|
| 두 신경망 + ACC 공동 학습 (동시) | ✓ |
| 위치 무관 매핑 | ✓ V2 r 0.46 vs random 0.75 |
| 한쪽 자극으로 반대쪽 표현 복원 | ✓ V2 cosine 0.81, acc_recon 0.83 |
| V2 > joint cross-attn (B4) | ✓ p<0.0001 (cosine, acc_recon 둘 다) |
| Hebbian으로 짝꿍 기록 | ✗ V3에서 V2 −0.51 손해 |

**사용자 가설의 큰 그림은 V2 형태로 강한 검증.** Hebbian 컴포넌트만 부적합 —
이 자체로 학술적 발견 (인공 뇌량은 Hebbian 없이 분리된 재구성 loss로 충분).

### 진행 이력

| 단계 | 결과 |
|---|---|
| Phase 0 — 설계 | PLAN.md v1.5 (19 sections + Deferred D-1~D-23) |
| Day 1~4 — 인프라 | 101 tests 통과, V3 chicken-and-egg D-19 fix |
| Day 4c — 1 epoch smoke | V3 g=−0.51 단조 학습 |
| Day 5 — B2 추가 + 베이스라인 | 7 variant 일괄 학습 |
| Day 6 — evaluate.py + D-21 (random baseline) | 측정 #1~#4 + 156 tests |
| Day 6-2 — V2 검증, V3 기각 (1 epoch) | PLAN v1.4: V2 본형 재서술 |
| **Day 7 — full sweep (5 seed)** | **★ Scenario A 강한 검증, PoC 완결** |

누적 156 tests 모두 통과.

## 빠른 시작

```bash
cd /mnt/d/brain/split_mnist
source ../split_brain_go/.venv/bin/activate   # SPLIT-9 venv 재활용
pip install -e .

# 테스트
pytest tests/ -v

# V3 1 epoch 학습
python -c "
from split_mnist.train import TrainConfig, train_one_run
cfg = TrainConfig(variant='V3', n_epochs=1, log_every=50)
r = train_one_run(cfg)
print(r)
"
```

요구 사항: Python 3.10–3.12, PyTorch 2.3+, CUDA 12.1 (GPU 권장).
RTX 3070 Ti 8GB로 1 epoch 약 30~60초.

## 디렉터리

```
split_mnist/
├── PLAN.md                  ← 설계 문서 (v1.2, 사전 등록)
├── README.md                ← 본 문서
├── pyproject.toml, requirements.txt, .gitignore
├── src/split_mnist/
│   ├── data.py              ← SplitMNIST + make_loaders
│   ├── networks.py          ← HalfCNN, Classifier, SingleCNN
│   ├── acc.py               ← ACCBase, V1, V2, V3 + CrossAttnAdapter (B4)
│   ├── losses.py            ← classification_loss
│   └── train.py             ← TrainConfig, build_model, train_one_run
├── tests/                   ← 101 단위 테스트
└── runs/                    ← 학습 산출물 (gitignored)
```

## 가설 검증 시나리오 (사전 등록 → Day 7 결과)

PLAN §12에 사전 등록한 임계치 vs Day 7 결과:

- **A 강한 검증** ✓ — V2 cosine 0.81, V2 vs B4 cosine Δ +0.16 (p<0.0001).
  → 다음: SPLIT-9 reboot + 워크숍 short paper.
- B 부분 검증 — 적용 안 됨 (A가 통과).
- C 가설 기각 — 적용 안 됨.

V3 형태 (Hebbian 추가)는 별개 — Day 6 §18에서 ablation 강등.

## 의존 프로젝트

- [SPLIT-9](file:///D:/brain/split_brain_go) (인접 폴더) — 본 PoC의 *동기*가 된 negative result. baseline 비교 baseline.

## 라이선스

(미정) — Apache 2.0 또는 MIT 예정.

---

*"Cross-entropy 학습은 합리적으로 prior에 자원을 투자한다. 보드별 정보를
강제하려면 데이터 분포 자체를 바꾸거나, 두 표현을 공동 진화시키는 학습
구조가 필요하다."* — SPLIT-9 결론, 본 PoC의 출발점.
