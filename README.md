# SPLIT-MNIST

🇰🇷 한국어 · [🇬🇧 English](README.en.md) · [🇨🇳 中文](README.zh.md) · [🇯🇵 日本語](README.ja.md)

*MNIST를 좌/우로 갈라 두 작은 CNN과 그 사이의 학습되는 인공 뇌량을 동시에 키웠을 때, 한쪽 자극만으로 반대쪽 표현을 복원할 수 있는지 검증한 PoC.*

앞 프로젝트 SPLIT-9(사후 어댑터)가 충실성의 구조적 천장에 부딪혀 음성으로 끝난 뒤, 그 천장을 깨는 가설 — 두 망을 from scratch로 *동시* 학습하며 사이에 *학습되는 인공 뇌량(ACC)* 을 두는 것 — 을 가장 작은 toy(좌/우로 가른 MNIST)에서 검증했다.
본 가설 본형(V2, 분리된 재구성 loss)은 한쪽 hidden을 ablation해도 반대쪽을 cosine 0.81 / 분류 0.83까지 복원했다 — 사후 어댑터 baseline(B4) 대비 cosine +0.16, acc_recon +0.11 (둘 다 p<0.0001). 사전 등록 임계 6개 중 5개 통과.
한편 신경과학적으로 자연스러운 Hebbian(V3)은 ML 구현에선 *잡음*으로 작용했다(V2 대비 cosine −0.51) — 그 자체로 하나의 발견. 메커니즘이 toy에서 동작함을 정량 입증한 **양성** 결과다.

**헤드라인:** V2 cosine 0.81 / acc_recon 0.83 · vs B4 +0.16 / +0.11 (p<0.0001) · 위치 무관 r 0.46 vs random 0.75 · 사전등록 5/6 통과 (Scenario A 강한 검증).

**[상세 보기 → 방법·측정 4종·표 (한/영)](https://halmoneysonmat.github.io/split-mnist/)**

---

### 연구 라인업

SPLIT-9의 음성이 이 PoC를 동기화했고, 이 양성이 다음 단계로 이어진다 — 두 망을 처음부터 *함께* 키우는 co-developed twins.

[SPLIT-9](https://github.com/HalmoneySonmat/split-9) → **SPLIT-MNIST** → [SPLIT-MAZE](https://github.com/HalmoneySonmat/split-maze)

재현 방법·스택·참고문헌은 상세 페이지에 있다. 테스트: `pytest tests/ -v` (~156개).

---

이 프로젝트를 만드는 데 Claude를 사용했다.
