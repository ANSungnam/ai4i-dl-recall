# -*- coding: utf-8 -*-
# AI4I 2020 예지보전 — 딥러닝 고장 분류
#
# 흐름: 데이터 확인 → 전처리 → 분할(80/20) → 표준화 → 학습 → 평가 → 새 데이터 예측
#
# 설계 원칙
#   - 입력은 CSV의 센서 특성 6개만 사용 (계산으로 만든 열 없음)
#   - 증강(class_weight)·임계값 조정 없음, 판정은 고장 확률 0.5 기준
#   - 출력은 고장 여부 + 고장유형 5개(멀티태스크), 고장유형은 결론으로 함께 제시
#
# 결과 해석 기준 (테스트셋 1,995건)
#   기준선: 전부 '정상'으로 찍어도 정확도 0.967 (고장 비율 3.3%)
#   상한  : 공구 마모 고장(TWF)은 200~240분 사이 무작위로 발생해 입력만으로는 예측 불가
#           → 테스트셋의 TWF 단독 11건을 빼고 모두 맞혀도 정확도 0.9945

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tf.keras.utils.set_random_seed(0)       # 가중치 초기값·데이터 섞기 순서 고정 → 실행마다 같은 결과

DIR = Path('~/ws/robot')
TARGET_ACC = 0.995                      # 목표 정확도

# ──────────────────────────────────────────────────────────
# [데이터 확인] 14개 열: UDI, Product ID, Type, 온도 2개, 회전수, 토크, 공구 마모,
#               Machine failure(정답), TWF·HDF·PWF·OSF·RNF(고장유형)
# ──────────────────────────────────────────────────────────
data = pd.read_csv(DIR / 'ai4i2020.csv', encoding='utf-8-sig')
pd.set_option('display.max_columns', None)     # 열이 생략(...)되지 않게
pd.set_option('display.width', 200)
print(data.head())
print('데이터 크기(행, 열):', data.shape)

# ──────────────────────────────────────────────────────────
# [전처리 1] 라벨 모순 행 제거
# 고장유형 중 하나라도 1이면 Machine failure도 1이어야 하는데, 이것이 어긋난 27행은
# 어느 표시가 맞는지 알 수 없어 제거 → 9,973행
# ──────────────────────────────────────────────────────────
type_names = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']   # 공구 마모 / 방열 / 전력 / 과부하 / 무작위
flag_any = data[type_names].any(axis=1)
data = data[flag_any == (data['Machine failure'] == 1)].reset_index(drop=True)

# ──────────────────────────────────────────────────────────
# [전처리 2] 입력 특성 선택
# 제외: UDI·Product ID(식별 번호), Machine failure·고장유형(정답 — 입력에 넣으면 누수)
# ──────────────────────────────────────────────────────────
feats = ['Type','Air temperature [K]','Process temperature [K]',
         'Rotational speed [rpm]','Torque [Nm]','Tool wear [min]']
X = data[feats].copy()

# [전처리 3] Type 인코딩 (H → 0, L → 1, M → 2) — encoder는 새 데이터 예측에 재사용
encoder = LabelEncoder()
X['Type'] = encoder.fit_transform(X['Type'])

# 정답 2종: 고장 여부(y)와 고장유형(Y_type, 출력 쪽에만 사용)
y = data['Machine failure'].to_numpy()        # 0 = 정상, 1 = 고장
Y_type = data[type_names].to_numpy()
target_names = np.array(['정상', '고장'])

# ──────────────────────────────────────────────────────────
# [분할] 훈련 80% / 테스트 20%
# stratify=y : 고장 330건이 양쪽에 같은 비율(3.3%)로 들어가게 함
# ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, T_train, T_test = train_test_split(
    X.to_numpy(), y, Y_type, test_size=0.2, random_state=0, stratify=y)

# ──────────────────────────────────────────────────────────
# [표준화] 특성마다 평균 0, 표준편차 1로 맞춤 (온도 300, 회전수 1500 같은 단위 차이 제거)
# 기준은 훈련셋에서만 계산(fit), 테스트셋은 적용(transform)만 — 섞으면 데이터 누수
# ──────────────────────────────────────────────────────────
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(f'훈련 {len(X_train)} / 테스트 {len(X_test)}')
print(f'기준선(전부 정상) 테스트 정확도: {(y_test == 0).mean():.4f}')

# ──────────────────────────────────────────────────────────
# [학습] 멀티태스크 신경망 — 은닉층 64-64(ReLU) 하나에서 두 출력이 갈라짐
#   은닉층은 두 출력의 손실 합을 줄이도록 학습 → 고장 여부와 고장유형을 함께 배움
#   'fail'  : 고장 확률 1개 → 판정·평가에 사용
#   'types' : 고장유형별 확률 5개 → 고장 원인을 결론으로 제시
# validation_data는 에폭마다 테스트 성적을 기록만 함(그래프용) — 학습·조정에는 쓰지 않음
# ──────────────────────────────────────────────────────────
inputs = Input(shape=(X_train.shape[1],))
h = Dense(64, activation='relu')(inputs)
h = Dense(64, activation='relu')(h)
out_fail = Dense(1, activation='sigmoid', name='fail')(h)
out_types = Dense(len(type_names), activation='sigmoid', name='types')(h)
model = Model(inputs, [out_fail, out_types])
model.compile(optimizer='adam',
              loss={'fail': 'binary_crossentropy', 'types': 'binary_crossentropy'},
              metrics={'fail': ['accuracy']})
history = model.fit(X_train, {'fail': y_train, 'types': T_train}, epochs=50, verbose=0,
                    validation_data=(X_test, {'fail': y_test, 'types': T_test}))

# ──────────────────────────────────────────────────────────
# [평가] 테스트셋 — 확률 0.5 이상이면 고장
#   정확도 = (TP+TN) / 전체   : 전체 판정 중 맞힌 비율
#   정밀도 = TP / (TP+FP)     : 고장 판정 중 진짜 고장 (낮으면 과검)
#   재현율 = TP / (TP+FN)     : 실제 고장 중 잡아낸 비율 (낮으면 미검)
# ──────────────────────────────────────────────────────────
p_fail, p_types = model.predict(X_test, verbose=0)
y_pred = (p_fail.ravel() >= 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print('\n[고장 여부 평가] 테스트셋')
print('1. 정확도(Accuracy) :', f'{acc:.4f}')
print('2. 정밀도(Precision):', f'{precision_score(y_test, y_pred, zero_division=0):.4f}')
print('3. 재현율(Recall)   :', f'{recall_score(y_test, y_pred):.4f}')
print(f'   혼동행렬: 맞힌 고장 {tp} / 놓친 고장(미검) {fn} / 오탐(과검) {fp} / 맞힌 정상 {tn}')
print('   목표 정확도', TARGET_ACC, '→', '달성' if acc >= TARGET_ACC else '미달')

# 결론: 고장유형별 성적
t_pred = (p_types >= 0.5).astype(int)
print('\n[고장유형별 결론] 테스트셋')
for i, name in enumerate(type_names):
    n_true = int(T_test[:, i].sum())
    n_hit = int(((T_test[:, i] == 1) & (t_pred[:, i] == 1)).sum())
    n_false = int(((T_test[:, i] == 0) & (t_pred[:, i] == 1)).sum())
    print(f'  {name}: 실제 {n_true:>2}건 중 {n_hit:>2}건 맞힘, 오탐 {n_false}건')

# Keras 내장 평가 — loss(전체), fail_loss·types_loss(출력별), fail_accuracy(고장 여부 정확도)
print('\n훈련 세트 평가:', model.evaluate(X_train, {'fail': y_train, 'types': T_train}, verbose=0, return_dict=True))
print('테스트 세트 평가:', model.evaluate(X_test, {'fail': y_test, 'types': T_test}, verbose=0, return_dict=True))

# 학습 곡선 (고장 여부 출력) — 훈련 손실만 내려가고 테스트 손실이 올라가면 과적합
epochs = np.arange(1, len(history.history['fail_loss']) + 1)

plt.figure(figsize=(7, 4))
plt.plot(
    epochs, history.history['fail_loss'],
    label='Training loss'
)
plt.plot(
    epochs, history.history['val_fail_loss'],
    label='Test loss'
)
plt.plot(
    epochs, history.history['fail_accuracy'],
    label='Training acc'
)
plt.plot(
    epochs, history.history['val_fail_accuracy'],
    label='Test acc'
)
plt.xlabel('Epoch')
plt.legend()
plt.show()

# ──────────────────────────────────────────────────────────
# [예측] 새 데이터 1건 — 순서: Type, 공기 온도, 공정 온도, 회전 속도, 토크, 공구 마모
# 샘플 1개도 2차원 (1, 6) 형태, 학습 때의 encoder·scaler를 그대로 사용(새로 fit 금지)
# ──────────────────────────────────────────────────────────
X_new = np.array([[
    encoder.transform(['M'])[0],        # 'M' → 2
    298.1, 308.6, 1551, 42.8, 0
]])
X_new = sc.transform(X_new)

pred_prob, type_prob = model.predict(X_new, verbose=0)
pred = (pred_prob.ravel() >= 0.5).astype(int)
print('\n예측 확률 [고장]:', pred_prob)
print('고장유형별 확률', type_names, ':', type_prob.round(4))
print('예측 고장유형:', [n for n, v in zip(type_names, type_prob.ravel()) if v >= 0.5] or '없음')
print('예측 번호:', pred)
print('예측 결과:', target_names[pred])
