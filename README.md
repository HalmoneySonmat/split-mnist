# SPLIT-MNIST

🇰🇷 한국어 · [🇬🇧 English](README.en.md) · [🇨🇳 中文](README.zh.md) · [🇯🇵 日本語](README.ja.md)

*MNIST를 좌/우로 갈라 두 작은 CNN과 그 사이의 학습되는 인공 뇌량을 *동시에* 키웠을 때, 한쪽 자극만으로 반대쪽 표현을 정말 복원할 수 있는지 검증한 PoC.*

> **TL;DR.** 인접 프로젝트 [SPLIT-9](../split_brain_go) 가 *post-hoc 어댑터*
> 가 자연어 설명의 충실성에서 구조적 천장에 부딪힌다는 negative result로
> 끝났다. 본 PoC는 그 한계를 푸는 *공동 학습 + 분리된 학습 신호 ACC*
> 가설이 동작하는지를 가장 작은 toy로 검증한다. MNIST 28×28을 좌/우
> 14×28로 갈라 두 작은 CNN과 그 사이의 인공 뇌량(ACC)을 *모두 from
> scratch로 같이* 학습. 5 seed × 5 epoch × 7 variant full sweep 결과,
> 본 가설 본형(V2)은 한쪽 hidden을 mean ablation해도 **cosine 0.81,
> 분류 정확도 0.83**까지 반대쪽을 복원했다. joint cross-attention
> baseline(B4) 대비 cosine **+0.16**, acc_recon **+0.11** (둘 다
> p<0.0001). W의 위치 무관성도 확인 (r=0.46 vs random 0.75). 사전 등록
> 임계 6개 중 5개 통과 — **Scenario A 강한 검증**. 한편 사용자 직관에
> 충실했던 Hebbian 컴포넌트(V3)는 ML 구현에서 W_learned와 자원 경쟁해
> *잡음*으로 작용 (V3 vs V2 cosine −0.51, p<0.0001) — 이 자체가 학술적
> 발견.

---

## 왜 이걸 만들었나

Sperry와 Gazzaniga의 분리뇌 실험 — 좌·우반구를 잇는 corpus callosum을
절단한 환자에게 우반구만 볼 수 있는 자리에 "걸어가시오" 카드를 보여주면
환자는 일어나 걷는다. "왜 걷냐?" 물으면 좌반구는 *진짜 이유*를 못 본
채로도 *그럴듯한 이야기*를 만들어 낸다 — 거짓말이 아니라 진짜로
*그렇다고 믿는다*. Gazzaniga가 *좌반구 통역사* 라고 부른 모듈.
유창하고, 일관되고, 자주 틀린다.

현대 AI는 이 구조를 의도적으로 반복한다. 시각 인코더, 로봇 정책, 결정
모델 같은 비언어 네트워크 옆에 LLM을 붙여 "지금 무슨 일인지" 자연어로
설명하게 한다. RLHF chain-of-thought, 비전-언어 어시스턴트, "설명
가능한" RL 에이전트 — 모두 같은 패턴. 그런데 LLM이 *진짜로 상위 신호를
번역*하는지, 아니면 *통계적으로 그럴듯한 텍스트만 뱉는지*는 대체로
검증되지 않는다.

인접 프로젝트 SPLIT-9가 이 질문을 9×9 바둑 + TinyLlama로 풀려
했다. 결과는 *Cross-entropy 학습이 토큰 엔트로피에 따라 prior에 자원을
투자한다* — 어댑터가 학습한 95%가 도메인 prior고, 5%만이 보드별
신호였다. 정성 검사 10개 중 0개가 정답 좌표를 짚었다. *Post-hoc 어댑터의
구조적 천장.*

본 PoC는 그 천장을 깨려는 다음 가설을 검증한다:

> 두 신경망을 *동시에 from scratch*로 학습시키며 그 사이에 *위치 무관
> coactivation 매핑*을 배우는 학습 단위(ACC)를 두면, 한쪽 자극만으로도
> 반대쪽 표현을 복원할 수 있다.

이걸 가장 작고 깨끗한 toy — 분리된 MNIST — 에서 정량 검증한다.

---

## 무엇을 만들었나

```
SPLIT-MNIST
├── MNIST 28×28
│       │
│       ├─ 좌 14×28 → 좌 CNN ─────────┐
│       └─ 우 14×28 → 우 CNN ─────────┤
│                                     │
│       ┌── 인공 뇌량 (ACC) ───────────┤   ← 학습 신호 분리:
│       │   W ∈ ℝ^(64×64)             │     분류 loss는 ACC 안 잡음
│       │   ĥ_R = h_L @ Wᵀ            │     ACC만의 loss로 학습
│       │   ĥ_L = h_R @ W             │
│       └── 변형: V1 / V2 / V3 ───────┘
│                                     │
│       cat[h_L, h_R] → 분류기 ───────┘
│                                  │
│                                  ▼
│                            0~9 분류 (γ 정책)
```

핵심 요소 — 모두 *from scratch 공동 학습*. 학습 *시간/데이터*는 같이,
학습 *gradient 경로*만 분리.

**ACC 변형 4가지** (학습 신호 차이만):
- **V1** Hebbian only — 사용자 원초 직관 ("같이 켜지면 짝꿍")
- **V2** ★ 재구성 only — backprop으로 한쪽→반대쪽 매핑 학습 (본 가설 본형)
- **V3** Hebbian + 재구성 — 두 가지 결합
- **B4** joint cross-attention — 분류 loss로만 학습 (post-hoc adapter baseline, SPLIT-9 패턴 본 toy 재현)

추가 베이스라인 — B1 (통합 CNN, 천장), B2(a)(b) (independent / left-only),
B3 (raw concat). 총 7+ variant.

**데이터.** 표준 MNIST 60k train(50k+10k val) + 10k test. 좌 14열 / 우
14열, 중복 없음. Normalize만.

**학습.** AdamW lr=1e-3, batch 128, 5 epoch, RTX 3070 Ti에서 ~28초/학습.
total 35 학습 = 19.5분.

---

## 무엇을 측정했나

### 1. Task accuracy (sanity)

5 epoch 후 모든 variant가 98.5~98.9% 사이로 천장(B1 98.72%)에 거의
따라잡음. **분류 자체에서는 ACC 디자인 차이 안 보임** — 측정 #2가
ACC variant를 *변별하는 결정적 도구*.

### 2. Cross-activation faithfulness ★

우 hidden을 mean ablation. ACC가 좌 hidden만으로 우 hidden 추정. 진짜 우
hidden과 cosine 비교 + 추정본을 분류기에 넣어 정확도 비교.

| variant | cosine | acc_real | acc_ablated | acc_recon | Δ(rec−abl) |
|---|---:|---:|---:|---:|---:|
| **V2** ★ | **0.814 ± 0.007** | 0.985 | 0.756 ± 0.029 | **0.829 ± 0.043** | **+0.073** |
| B4 | 0.653 ± 0.026 | 0.986 | 0.740 ± 0.040 | 0.722 ± 0.039 | −0.018 |
| V3 | 0.303 ± 0.034 | 0.985 | 0.759 ± 0.037 | 0.681 ± 0.052 | −0.079 |
| V1 | 0.273 ± 0.107 | 0.985 | 0.737 ± 0.056 | 0.418 ± 0.080 | −0.319 |

**V2가 *유일하게* `acc_recon > acc_ablated`** — ACC 복원본이 진짜로 분류
도움. 다른 모든 variant는 random ablation보다 못 복원.

특히 **B4**의 Δ가 음수(−0.018) — *joint 학습 cross-attention이 random
보다도 못 복원*. SPLIT-9 negative 패턴의 본 toy 재현. "post-hoc adapter
는 분류만 잡고 복원은 못 한다"가 통계적 확인.

### 3. Causal coupling — toy 한계

좌 hidden에 ε∈{0, 0.1, 0.25, 0.5, 1.0, 2.0} 가우시안 noise. 모든 variant
에서 IAS@0.5 ≈ 0. **MNIST 분리뇌가 너무 쉬움** (좌 14×28만으로도 95%+
분류 가능 → 우는 redundant → 좌 noise도 무영향). 측정 #3은 본 toy에서
정보적이지 않음. 더 어려운 task 필요 (D-22 deferred).

### 4. Position invariance — D-21 random baseline 기반

5 seed의 ACC W를 Procrustes 정렬해 평균 상관 r. 64×64 random isotropic
matrix는 그 자체로 r ≈ 0.75 (random matrix theory). 그 위에서 학습된 W
는 어떤가:

| variant | r_trained | Δ vs random (0.7481) | 판정 |
|---|---:|---:|---|
| **V2** | **0.4646** | **−0.2835** | ✓ A (위치 무관 강함) |
| V1 | 0.4105 | −0.3376 | ✓ A |
| V3 | 0.5600 | −0.1881 | ✓ A (살짝) |
| B4 | 0.4055 | −0.3426 | ✓ A |

V2가 임계 0.15의 약 2배 통과. **5 seed가 *서로 다른 짝꿍 패턴* 학습 —
위치 무관 가설 강한 검증**. 흥미롭게 V3가 가장 *덜* 위치 무관 (Hebbian이
seed간 비슷한 통계에 수렴 시도 = Hebbian 잡음의 정체).

### Paired bootstrap — 통계적 결정타

```
V2 vs B4 (★ central hypothesis test):
  cosine     Δ = +0.1606 ± 0.0306   p = 0.0000   임계 ≥ 0.15  ✓
  acc_recon  Δ = +0.1066 ± 0.0303   p = 0.0000   임계 ≥ 0.05  ✓

V3 vs V2 (Hebbian 추가 효과):
  cosine     Δ = −0.5111 ± 0.0325   p = 0.0000  ← Hebbian 잡음 통계적 확정
  acc_recon  Δ = −0.1484 ± 0.0926   p = 0.0000
```

**둘 다 Scenario A 임계 통과 + p = 0.0000.** 5 seed 모두 일관되게 V2
우월 — 우연 거의 0.

---

## 시나리오 판정 — Scenario A 강한 검증

PLAN §12 사전 등록 임계 6개:

| 측정 | 임계 | 결과 | 통과 |
|---|---|---|---|
| #1 V2 ≥ B1 − 5%p | 92.4% | 98.5% | ✓ |
| #2 V2 cosine ≥ 0.7 | 0.7 | 0.81 | ✓ |
| #2 V2 vs B4 cosine Δ ≥ 0.15 | 0.15 | +0.16 (p<0.0001) | ✓ |
| #2 V2 vs B4 acc_recon Δ ≥ 5%p | 0.05 | +0.11 (p<0.0001) | ✓ |
| #3 V2 인과 결합 | 5%p | toy 한계 | ⚠ |
| #4 V2 r ≤ random − 0.15 | ≤ 0.60 | 0.46 | ✓ |

**5/6 통과**. 측정 #3만 toy 한계로 약함 (PLAN §18.5에서 사전 인정됨,
즉 결과 보기 *전*에 발견).

---

## 이게 무슨 의미인가

**두 신경망을 공동 학습하면서 그 사이에 *학습되는 인공 뇌량*을 두면,
post-hoc 어댑터(SPLIT-9 패턴)가 풀지 못한 cross-activation faithfulness
문제를 해결할 수 있다.** 본 PoC가 그 *원리적 가능성*을 가장 작은 toy
에서 정량적으로 입증.

핵심은 **학습 신호의 분리**(γ 정책): ACC가 분류 loss로부터 직접 backprop
을 받지 않고 자기만의 *재구성 loss*로 학습하면, ACC가 진짜 매핑을 학습.
joint cross-attention(B4)은 분류 loss에 잡혀 *분류만 잘 하는 매핑*을
학습 — 결과적으로 random ablation보다 못 복원. SPLIT-9 negative의 *원인*
이 본 toy에서 확인됨.

또 한 가지 — **Hebbian은 ML 구현에서 부적합**. V3에서 W_hebbian이 임의
outer product에 누적되어 W_learned가 *그 잡음 상쇄*에 자원을 빼앗김
(η/W_max sweep으로 확정한 *구조적 한계*). V3가 V2보다 cosine 0.51 손해.
신경과학적으로 충실한 직관이 ML에서는 *역효과* — **인공 뇌량은 Hebbian
없이 분리된 재구성 loss + γ 정책만으로 충분**.

이 발견은 Hebbian을 직접 도입한 기존 ML 작업들 (Hybrid SOM 2022,
HeLa-Mem 2025, Hebbian Fast Weights in ViT 2025)과 다른 결론. 본 PoC는
"공동 학습 환경 + 분류 loss와 분리된 재구성 loss" 조합이 핵심이고,
Hebbian은 옵션 아닌 부담일 수 있음을 시사.

---

## 한계 (PoC 범위)

본 결과의 일반화에 신중해야 할 4가지:

1. **MNIST**: 두 modality가 동질적(둘 다 작은 CNN, 둘 다 이미지 절반).
   진짜 이질적 환경(예: 결정자 + LLM)에서는 동작 양상이 다를 수 있음.
2. **너무 쉬운 task**: 좌만으로 95%+ 분류 가능. 측정 #3 (인과 결합)이
   정보적이지 않음.
3. **"해석" 검증 안 됨**: PoC는 *수치적 복원*만 확인. SPLIT-9의
   *자연어 설명 충실성*은 별도 차원.
4. **단방향만**: 사용자 가설의 full form (양방향 뇌량 — LLM 설명이
   다음 결정에 영향)은 검증 안 됨.

---

## 향후 연구 — SPLIT-9 reboot

본 PoC가 가능성을 보였으므로 다음 자연스러운 단계:

1. **SPLIT-9 환경에 V2 형태 ACC 적용** — 9×9 Go-Net + 동결 LLM 사이에
   *공동 학습되는 V2 ACC*. SPLIT-9의 negative result(95% 도메인 prior,
   5% 보드별 신호)가 *해소*되는지.
2. **양방향 뇌량** — LLM 설명이 Go-Net 다음 결정에 영향. 사용자 가설의
   full form. 분리뇌 → 정상 뇌로의 전환.
3. **D-22**: 측정 #3이 정보적인 더 어려운 toy/task.
4. **D-23**: V3' "Hebbian opt-in" (`W = tanh(α)·W_h + W_l`) — Hebbian이
   특정 조건에서 도움 주는지.

본 PoC는 그 길의 *베이스 캠프* — 메커니즘이 toy에서 동작함을 정량 입증.

---

## 재현 방법

```bash
# 환경 (RTX 3070 Ti / WSL2 Ubuntu 검증, 8 GB GPU 1대)
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 테스트
pytest tests/ -v   # 156개

# Day 7 full sweep — 5 seed × 5 epoch × 7 variant (~20분)
python scripts/run_full_sweep.py
```

결과는 `runs/sweep_results.json`에 저장. 콘솔에 표 4개 + paired bootstrap.

요구 사항: Python 3.10–3.12, PyTorch 2.3+, CUDA 12.1, NumPy, SciPy.

---

## 디렉터리

```
split_mnist/
├── PLAN.md                  ← 사전 등록 설계 문서 (v1.5, 19 sections)
├── README.md                ← 본 문서 (한국어 메인)
├── README.en.md             ← English
├── README.ja.md             ← 日本語
├── LICENSE                  ← Apache 2.0
├── pyproject.toml, requirements.txt, .gitignore
├── src/split_mnist/
│   ├── data.py              ← SplitMNIST + make_loaders
│   ├── networks.py          ← HalfCNN, Classifier, IndependentClassifiers, SingleCNN
│   ├── acc.py               ← ACCBase, V1/V2/V3 + CrossAttnAdapter (B4)
│   ├── losses.py            ← classification_loss
│   ├── train.py             ← TrainConfig, build_model, train_one_run, evaluate_left_only
│   └── evaluate.py          ← measure_task_accuracy, _cross_activation, _causal_coupling, _position_invariance, _random_baseline
├── scripts/
│   ├── train_baselines.py   ← 7 variant 1 seed × 1 epoch (smoke)
│   ├── evaluate_baselines.py← 7 variant 1 seed × 1 epoch + 측정 #2/#3
│   └── run_full_sweep.py    ← Day 7 본 게임: 5 seed × 5 epoch + 측정 #1~#4 + paired bootstrap
├── tests/                   ← 156 단위 테스트
└── runs/                    ← 학습 산출물 (gitignored)
    └── sweep_results.json   ← Day 7 결과
```

---

## 스택

Python 3.10–3.12 · PyTorch 2.3+ · NumPy · SciPy · pytest · 8 GB 소비자용
GPU 1대. CUDA 12.1.

---

## 참고 문헌

* Alayrac et al., *Flamingo: a Visual Language Model for Few-Shot Learning*, NeurIPS 2022.
* Haxby et al., *Hyperalignment: Modeling shared information encoded in idiosyncratic cortical topographies*, eLife 2020.
* Lambon Ralph et al., *The Roles of Left Versus Right Anterior Temporal Lobes in Semantic Memory*, Cerebral Cortex 2018.
* Koganemaru et al., *Paired associative stimulation across hemispheres via corpus callosum* (interhemispheric Hebbian PAS).
* Innocenti & Price, *Exuberance in the development of cortical networks*, Nature Reviews Neuroscience 2005.
* Gazzaniga, *The Bisected Brain*, Appleton-Century-Crofts, 1970.
* Gazzaniga, *The Consciousness Instinct*, Farrar Straus Giroux, 2018.
* Turpin et al., *Language Models Don't Always Say What They Think*, NeurIPS 2023.
* Atanasova et al., *Faithfulness Tests for Natural Language Explanations*, ACL 2023.
* Conmy et al., *Towards Automated Circuit Discovery for Mechanistic Interpretability*, NeurIPS 2023.
* (인접 폴더 SPLIT-9 — post-hoc adapter negative result)

---

## 상태

PoC 완결. v0.3-day7-scenarioA. Scenario A 강한 검증.
워크숍 short paper 작성 가능. 다음 자연스러운 단계는 SPLIT-9 reboot
(9×9 바둑 + LLM에 V2 형태 ACC 적용).

---

## 라이선스

Apache License 2.0. [LICENSE](LICENSE) 참조.

```
Copyright 2026 namdo

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

---

*"두 신경망을 동시에 키울 때, 그 사이의 다리도 같이 키워라. 그 다리에게
는 자기만의 일을 시켜라. 그러면 한쪽이 본 것을 다른 쪽이 깨운다."*
— 본 PoC의 결론을 한 문장으로.

---

*재밌는 프로젝트였다.*
