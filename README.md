# AI4I 2020 예지보전 딥러닝 (Predictive Maintenance)

AI4I 2020 데이터셋으로 기계 고장을 예측하는 Keras 신경망 분류기.
불균형 데이터(고장 3.3%)에서 **재현율(Recall)** 을 목표 지표로 삼아,
class_weight와 검증셋 기반 교정 루프로 고장 탐지율을 끌어올린다.

## 파이프라인

```
[전처리]    라벨 노이즈 제거(모순 27행) → 특성 선택(누수 열 제외) → 인코딩
[분할]      훈련 60 / 검증 20 / 테스트 20  (stratify로 고장 비율 유지)
[표준화]    훈련셋 기준으로만 fit — 데이터 누수 방지
[학습+교정] class_weight 후보(COST_RATIO)를 검증 재현율로 교정 (루프)
[최종 확인] 테스트셋 1회 측정 — 결과를 보고 재조정하지 않음
```

핵심 설계 결정과 이유는 전부 `ai4i_dl_recall.py`의 주석에 기록되어 있다.

- 고장 유형 열(TWF·HDF·PWF·OSF·RNF)은 타깃의 세부 라벨이라 **입력에서 제외** (데이터 누수)
- 단, 라벨 정합성 검사에는 사용: 플래그와 Machine failure가 모순인 **27행을 노이즈로 제거**
- `stratify` = 분할(채점)의 공정성, `class_weight` = 학습의 공정성 — 역할이 달라 둘 다 사용
- 역빈도 가중치(≈1:29)는 고장 손실 비중을 50%로 만드는 **출발점**이며, 최적점은
  비용 비율(FN비용/FP비용)에 따라 검증셋 루프로 탐색
- `tf.keras.utils.set_random_seed(0)` 으로 실행 간 재현성 고정

## 데이터

[UCI — AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i) (CC BY 4.0)

`ai4i2020.csv` 를 내려받아 코드의 경로(`~/ws/robot/ai4i2020.csv`)에 두거나, `read_csv` 경로를 수정한다.

## 실행

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
python ai4i_dl_recall.py
```

## 결과 (seed 0, 테스트셋 1,995건)

| 지표 | 값 |
|---|---|
| 검증 재현율 (교정 기준) | 0.909 |
| 테스트 재현율 | **0.833** |
| 테스트 정밀도 | 0.440 |
| F1 | 0.576 |
| 놓친 고장(FN) / 오탐(FP) | 11건 / 70건 |

보정 없는 기본 학습(재현율 ≈ 0.57~0.69) 대비 놓치는 고장이 크게 줄었다.
정확도(≈0.96)는 불균형 데이터에서 '전부 정상' 예측으로도 나오는 수치라 판단 지표로 쓰지 않는다.

## 참고

- S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications," AI4I 2020
