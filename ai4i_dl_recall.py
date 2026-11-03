# -*- coding: utf-8 -*-
# AI4I 2020 예지보전 — 딥러닝 고장 분류 (정확도 99.5% 목표, 증강 없음)
#
# 전체 흐름
#   [전처리]  라벨 모순 행 제거 → 입력 특성 선택 → 문자(Type)를 숫자로 인코딩
#   [분할]    훈련 80% / 테스트 20%
#   [표준화]  훈련셋 기준으로만 평균·표준편차를 배워서 양쪽에 적용
#   [학습]    가중치·증강 없이 원래 데이터 분포 그대로 1회 학습
#   [평가]    테스트셋으로 정확도·정밀도·재현율 측정
#   [예측]    새 데이터 1건의 고장 여부 판정
#
# 결과를 읽을 때의 기준
#   기준선: 전부 '정상'이라고 찍어도 정확도 ≈ 96.7% (고장이 3.3%뿐이라서)
#           → 모델 정확도는 97%가 아니라 이 96.7% 대비 얼마나 올렸는지로 봐야 함
#   상한  : 공구 마모 고장(TWF)은 200~240분 사이 무작위로 발생해서
#           입력 6개로는 원리상 맞힐 수 없음 → 정확도 이론 상한 ≈ 99.57%

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 재현성 고정: 가중치 초기값·데이터 섞기 순서를 시드 0으로 고정 → 매번 같은 결과
tf.keras.utils.set_random_seed(0)

DIR = Path('~/ws/robot')
TARGET_ACC = 0.995                      # 목표: 테스트 정확도 99.5% 이상

# ──────────────────────────────────────────────────────────
# [전처리 1] 라벨 모순 행 제거
# 규칙: 고장 유형(TWF·HDF·PWF·OSF·RNF) 중 하나라도 1이면 Machine failure도 1이어야 함
# 이 규칙이 어긋난 27행(유형 표시 있는데 고장 0: 18행 / 유형 없는데 고장 1: 9행)은
# 어느 쪽이 맞는지 알 수 없어서 고치지 않고 뺌 → 9,973행 남음
# ──────────────────────────────────────────────────────────
data = pd.read_csv(DIR / 'ai4i2020.csv', encoding='utf-8-sig')
flag_any = data[['TWF','HDF','PWF','OSF','RNF']].any(axis=1)
data = data[flag_any == (data['Machine failure'] == 1)].reset_index(drop=True)

# ──────────────────────────────────────────────────────────
# [전처리 2] 입력 특성 선택
# 제외: UDI·Product ID(단순 식별 번호), TWF~RNF(고장 유형 = 정답이 새는 열)
# ──────────────────────────────────────────────────────────
feats = ['Type','Air temperature [K]','Process temperature [K]',
         'Rotational speed [rpm]','Torque [Nm]','Tool wear [min]']
X = data[feats].copy()

# [전처리 3] 인코딩: 품질 등급 문자를 숫자로 (알파벳순 H → 0, L → 1, M → 2)
# encoder는 변수로 저장해 두었다가 새 데이터 예측 때 같은 규칙으로 다시 씀
encoder = LabelEncoder()
X['Type'] = encoder.fit_transform(X['Type'])

y = data['Machine failure'].to_numpy()        # 정답: 0 = 정상, 1 = 고장
target_names = np.array(['정상', '고장'])     # 예측 번호 → 이름 변환용

# ──────────────────────────────────────────────────────────
# [분할] 훈련 80% / 테스트 20%
# 조정할 값(가중치·임계값)이 없어서 검증셋을 따로 두지 않음
# stratify=y : 훈련·테스트 양쪽에 고장 비율(3.3%)을 똑같이 맞춤
#              고장이 330건뿐이라 무작위로 나누면 테스트에 고장이 쏠리거나 모자랄 수 있음
# ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X.to_numpy(), y, test_size=0.2, random_state=0, stratify=y)

# ──────────────────────────────────────────────────────────
# [표준화] 각 특성을 평균 0, 표준편차 1로 맞춤 (온도 300, 회전수 1500처럼 단위 차이 제거)
# 평균·표준편차는 훈련셋에서만 계산(fit)하고, 테스트셋에는 그 기준을 적용만 함
# → 테스트셋까지 섞어서 fit하면 시험 문제를 미리 본 셈(데이터 누수)
# ──────────────────────────────────────────────────────────
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(f'훈련 {len(X_train)} / 테스트 {len(X_test)}')
print(f'기준선(전부 정상) 테스트 정확도: {(y_test == 0).mean():.4f}')

# ──────────────────────────────────────────────────────────
# [학습] 신경망 정의 → 학습 1회
# 은닉층 64-64(ReLU), 출력층 1개(sigmoid) = 고장 확률 0~1
# 가중치(class_weight)·증강을 쓰지 않음: 고장을 인위적으로 부풀리면
# 재현율은 오르지만 과검(양품을 고장으로 판정)이 늘어 정밀도가 떨어지기 때문
# validation_data : 에폭마다 테스트 성적을 기록만 함(그래프용), 학습·조정에는 안 쓰임
# ──────────────────────────────────────────────────────────
print(f'\n[학습] 가중치 없이 학습 (목표 정확도 ≥ {TARGET_ACC})')
model = Sequential([
    Input(shape=(6,)),                  # 입력 = 특성 6개
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')      # 출력 1개 = 고장 확률 (0~1)
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, verbose=0,
                    validation_data=(X_test, y_test))

# ──────────────────────────────────────────────────────────
# [평가] 테스트셋 지표 — 고장 확률 0.5 이상이면 고장(1)으로 판정
#   정확도 : 전체 판정 중 맞힌 비율            (TP+TN) / 전체
#   정밀도 : 고장이라고 판정한 것 중 진짜 고장   TP / (TP+FP)  → 낮으면 과검이 많음
#   재현율 : 실제 고장 중 고장으로 잡아낸 비율   TP / (TP+FN)  → 낮으면 미검이 많음
# ──────────────────────────────────────────────────────────
y_pred = (model.predict(X_test, verbose=0).ravel() >= 0.5).astype(int)
print('1. 정확도(Accuracy) :', f'{accuracy_score(y_test, y_pred):.4f}')
print('2. 정밀도(Precision):', f'{precision_score(y_test, y_pred, zero_division=0):.4f}')
print('3. 재현율(Recall)   :', f'{recall_score(y_test, y_pred):.4f}')
print('   목표 정확도', TARGET_ACC, '→', '달성' if accuracy_score(y_test, y_pred) >= TARGET_ACC else '미달')

# 모델 평가하기 (Keras 내장)
# 출력 순서: [손실, 정확도] — 정확도는 Keras도 0.5 기준으로 계산
print('훈련 세트 평가:', model.evaluate(X_train, y_train))
print('테스트 세트 평가:', model.evaluate(X_test, y_test))

# 학습 과정 그래프 — 훈련/테스트 손실이 함께 내려가면 정상,
# 훈련 손실만 계속 내려가고 테스트 손실이 올라가면 과적합 신호
epochs = np.arange(1, len(history.history['loss']) + 1)

plt.figure(figsize=(7, 4))
plt.plot(
    epochs, history.history['loss'],
    label='Training loss'
)
plt.plot(
    epochs, history.history['val_loss'],
    label='Validation loss'
)
plt.plot(
    epochs, history.history['accuracy'],
    label='Training acc'
)
plt.plot(
    epochs, history.history['val_accuracy'],
    label='Validation acc'
)
plt.xlabel('Epoch')
plt.legend()
plt.show()

# ──────────────────────────────────────────────────────────
# [예측] 새로운 데이터 1건
# encoder.transform(['M']) → array([2])
# encoder.transform(['M'])[0] → 2
# 순서: Type, 공기 온도, 공정 온도, 회전 속도, 토크, 공구 마모
# 샘플이 1개라도 2차원 (1, 6) 형태여야 함
# ──────────────────────────────────────────────────────────
X_new = np.array([[
    encoder.transform(['M'])[0],
    298.1, 308.6, 1551, 42.8, 0
]])

# 학습 때 쓴 스케일러로 표준화한 뒤 예측 (새로 fit 금지)
# 표준화를 빼면 298, 1551 같은 원래 값이 그대로 들어가 확률이 엉뚱하게 튐
X_new = sc.transform(X_new)
pred_prob = model.predict(X_new)
print('예측 확률 [고장]:', pred_prob)              # 출력 1개 = 고장 확률

pred = (pred_prob.ravel() >= 0.5).astype(int)       # 고장 확률 0.5 이상 → 고장(1)
print('예측 번호:', pred)
print('예측 결과:', target_names[pred])             # 0 → 정상, 1 → 고장
