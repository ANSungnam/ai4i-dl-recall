# AI4I 2020 예지보전 딥러닝 (Predictive Maintenance)

AI4I 2020 데이터셋으로 기계 고장을 예측하는 Keras 신경망 분류기.
불균형 데이터(고장 3.3%)에서 **정확도 99.5%** 를 목표로, 가중치·증강 없이
원래 데이터 분포 그대로 학습하고 정확도·정밀도·재현율을 함께 평가한다.

## 파이프라인

```
[전처리]  라벨 모순 27행 제거 → 특성 선택(누수 열 제외) → 인코딩
[분할]    훈련 80 / 테스트 20  (stratify로 고장 비율 유지)
[표준화]  훈련셋 기준으로만 fit — 데이터 누수 방지
[학습]    가중치·증강 없이 1회 학습
[평가]    테스트셋 정확도·정밀도·재현율 (고장 확률 0.5 기준)
[예측]    새 데이터 1건 판정 (학습 때 스케일러로 표준화 후)
```

각 단계의 이유는 `ai4i_dl_recall.py` 주석에 정리되어 있다.

- 고장 유형 열(TWF·HDF·PWF·OSF·RNF)은 정답의 세부 라벨이라 **입력에서 제외** (데이터 누수)
- 단, 라벨 정합성 검사에는 사용: 유형 표시와 Machine failure가 모순인 **27행 제거**
- **증강(class_weight) 미사용**: 고장을 부풀리면 재현율은 오르지만 과검이 늘어 정밀도가 떨어짐
- 조정할 값이 없어 검증셋을 두지 않고 80/20으로 분할
- `tf.keras.utils.set_random_seed(0)` 으로 실행 간 재현성 고정

## 데이터

[UCI — AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (CC BY 4.0)

`ai4i2020.csv` 를 내려받아 코드의 경로(`~/ws/robot/ai4i2020.csv`)에 두거나, `read_csv` 경로를 수정한다.

## 실행

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
python ai4i_dl_recall.py
```

## 결과 (seed 0, 테스트셋 1,995건)

| 지표 | 값 |
|---|---|
| 정확도 | **0.9835** |
| 정밀도 | 0.837 |
| 재현율 | 0.621 |
| 놓친 고장(FN) / 오탐(FP) | 25건 / 8건 |
| 목표 정확도 99.5% | 미달 |

결과를 읽는 기준:

| 구분 | 정확도 | 설명 |
|---|---|---|
| 기준선 | 0.967 | 전부 '정상'으로 찍어도 나오는 값 |
| 이 모델 | 0.9835 | |
| 이론 상한 | ≈ 0.9957 | 공구 마모 고장(TWF)은 200~240분 사이 무작위 발생이라 입력 6개로는 원리상 예측 불가 |

이전 버전(class_weight ≈ 29배 + 재현율 0.9 목표)은 재현율 0.833, 정밀도 0.440이었다.
증강을 빼자 과검이 줄어 정밀도·정확도가 올랐고, 대신 놓친 고장이 늘었다.

## 참고

- S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications," AI4I 2020, IEEE, pp. 69–74, doi: 10.1109/AI4I49448.2020.00023
  - 데이터 제작자도 신경망·SVM을 시험한 뒤 bagged trees를 택했고, 미검 비용을 오탐의 30배로 두어 재현율 86.7%, 정밀도 약 71%를 보고했다.
