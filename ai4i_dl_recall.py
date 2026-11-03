# -*- coding: utf-8 -*-
# AI4I 2020 예지보전 — 딥러닝 고장 분류 + 판정 보류 (5겹 교차 검증으로 보류 기준 결정)
#
# 흐름: 데이터 확인 → 전처리 → 분할(80/20) → 5겹 교차 검증으로 보류 기준 결정
#       → 최종 학습(훈련 80% 전체) → 테스트 20%로 1회 평가 → 새 데이터 예측
#
# 설계 원칙
#   - 입력은 CSV의 센서 특성 6개만 사용 (계산으로 만든 열 없음), 증강 없음
#   - 출력은 고장 여부 + 고장유형 5개(멀티태스크)
#   - 판정은 3가지: 불량 / 양품 / 판정 보류(사람 확인)
#       1단계: 고장 확률이 [하한, 상한] 사이면 보류 — 모델도 헷갈리는 샘플
#       2단계: 고장유형 출력 5개 각각 '확률이 기준 이상이면 보류' 후보를 두고,
#              어느 출력을 쓸지·기준값은 얼마인지를 코드가 교차 검증 결과로 선택
#              (사람이 특정 고장유형을 지정하지 않음 — 쓸모없는 출력은 '사용 안 함'으로 정해짐)
#   - 사람이 정하는 것: 보류 허용, 보류 상한(MAX_DEFER), 목표 정확도(TARGET_ACC), 후보값 범위
#   - 보류 기준은 훈련 80% 안의 교차 검증 예측으로만 정하고, 테스트 20%는 마지막 채점에만 사용
#
# 결과 해석 기준 (테스트셋 1,995건)
#   기준선: 전부 '정상'으로 찍어도 정확도 0.967 (고장 비율 3.3%)
#   상한  : 보류 없이 판정하면 TWF 단독 11건은 원리상 못 맞혀 정확도 최대 0.9945

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
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
TARGET_ACC = 0.995                      # 목표: 판정한 것의 정확도
MAX_DEFER = 0.10                        # 보류 비율 상한 (사람이 감당할 수 있는 양, 10%)

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
# 테스트 20%는 맨 마지막 채점에만 사용 — 학습·기준 결정에는 쓰지 않음
# ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, T_train, T_test = train_test_split(
    X.to_numpy(dtype=float), y, Y_type, test_size=0.2, random_state=0, stratify=y)
print(f'훈련 {len(X_train)} / 테스트 {len(X_test)}')
print(f'기준선(전부 정상) 테스트 정확도: {(y_test == 0).mean():.4f}')

# ──────────────────────────────────────────────────────────
# [모델 정의] 멀티태스크 신경망 — 은닉층 64-64(ReLU) 하나에서 두 출력이 갈라짐
#   'fail'  : 고장 확률 1개 → 판정에 사용
#   'types' : 고장유형별 확률 5개 → 보류 판단 후보, 고장 원인 제시
# 교차 검증 5번 + 최종 1번, 모두 6번 새로 만들어야 해서 함수로 둠
# ──────────────────────────────────────────────────────────
def build_model():
    inputs = Input(shape=(X_train.shape[1],))
    h = Dense(64, activation='relu')(inputs)
    h = Dense(64, activation='relu')(h)
    out_fail = Dense(1, activation='sigmoid', name='fail')(h)
    out_types = Dense(len(type_names), activation='sigmoid', name='types')(h)
    m = Model(inputs, [out_fail, out_types])
    m.compile(optimizer='adam',
              loss={'fail': 'binary_crossentropy', 'types': 'binary_crossentropy'},
              metrics={'fail': ['accuracy']})
    return m

# 판정: 1 = 불량, 0 = 양품, -1 = 보류
#   type_ths: 고장유형 5개 각각의 보류 기준 (None = 그 출력은 보류에 사용 안 함)
def decide(p_fail, p_types, lo, hi, type_ths):
    pred = (p_fail >= 0.5).astype(int)                 # 기본 판정 (확률 0.5 기준)
    defer = (p_fail > lo) & (p_fail < hi)              # 1단계: 고장 확률이 애매함
    for i, th in enumerate(type_ths):                  # 2단계: 고장유형 확률이 기준 이상
        if th is not None:
            defer |= (p_types[:, i] >= th)
    pred[defer] = -1
    return pred

# ──────────────────────────────────────────────────────────
# [5겹 교차 검증] 훈련 80%를 5조각으로 나눠, 매번 4조각 학습 → 1조각 예측 (5번 반복)
# 모든 훈련 샘플이 '학습에 안 쓴 모델'로 한 번씩 예측됨 → 이 예측으로 보류 기준을 정함
# 표준화도 조각마다 그 학습 부분으로만 fit (검증 조각 정보가 새지 않게)
# ──────────────────────────────────────────────────────────
print('\n[5겹 교차 검증] 보류 기준을 정할 예측 만드는 중')
oof_fail = np.zeros(len(X_train))                       # 검증 조각에서 나온 고장 확률 (out-of-fold)
oof_types = np.zeros((len(X_train), len(type_names)))   # 검증 조각에서 나온 고장유형별 확률
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for k, (i_fit, i_val) in enumerate(skf.split(X_train, y_train), start=1):
    sc_k = StandardScaler().fit(X_train[i_fit])
    m_k = build_model()
    m_k.fit(sc_k.transform(X_train[i_fit]), {'fail': y_train[i_fit], 'types': T_train[i_fit]},
            epochs=50, verbose=0)
    pf, pt = m_k.predict(sc_k.transform(X_train[i_val]), verbose=0)
    oof_fail[i_val], oof_types[i_val] = pf.ravel(), pt
    acc_k = accuracy_score(y_train[i_val], (pf.ravel() >= 0.5).astype(int))
    print(f'  {k}번째 조각: 학습 {len(i_fit)} / 검증 {len(i_val)}, 검증 정확도(보류 없음) {acc_k:.4f}')

# ──────────────────────────────────────────────────────────
# [보류 기준 결정] 교차 검증 예측으로 모든 조합을 비교 (1단계 구간 × 고장유형 5개 각각의 기준)
# 규칙: 보류 비율 ≤ MAX_DEFER 인 조합 중
#   (1) 판정한 것 중 놓친 불량(미검)이 가장 적은 조합 — 불량을 양품으로 내보내는 게 가장 큰 손해
#   (2) 미검이 같으면 → 판정한 것의 전체 오류가 적은 조합
#   (3) 그것도 같으면 → 보류가 적은 조합
# ──────────────────────────────────────────────────────────
import itertools
LOS, HIS = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02], [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
TH_OPTIONS = [None, 0.05, 0.02, 0.01]                   # 고장유형 출력별 보류 기준 후보
base_pred = (oof_fail >= 0.5).astype(int)
correct = base_pred == y_train
missed = (base_pred == 0) & (y_train == 1)               # 보류 안 하면 놓치는 불량
band = {(lo, hi): (oof_fail > lo) & (oof_fail < hi) for lo in LOS for hi in HIS}
type_mask = {(i, th): oof_types[:, i] >= th
             for i in range(len(type_names)) for th in TH_OPTIONS if th is not None}
best = None
for type_ths in itertools.product(TH_OPTIONS, repeat=len(type_names)):
    t_defer = np.zeros(len(X_train), bool)
    for i, th in enumerate(type_ths):
        if th is not None:
            t_defer |= type_mask[(i, th)]
    for (lo, hi), b_defer in band.items():
        defer = b_defer | t_defer
        rate = defer.mean()
        if rate > MAX_DEFER or rate == 1:
            continue
        acc = correct[~defer].mean()
        n_missed = int((missed & ~defer).sum())             # 판정한 것 중 놓친 불량
        n_err = int((~correct & ~defer).sum())              # 판정한 것 중 전체 오류
        key = (-n_missed, -n_err, -rate)
        if best is None or key > best[0]:
            best = (key, lo, hi, type_ths, rate, acc)
_, LO, HI, TYPE_THS, cv_defer, cv_acc = best
print(f'\n[코드가 정한 보류 기준] (교차 검증 예측으로 선택)')
print(f'  1단계: 고장 확률 {LO} ~ {HI} 사이면 보류')
for name, th in zip(type_names, TYPE_THS):
    print(f'  2단계: {name} 출력 → ' + (f'확률 ≥ {th} 이면 보류' if th is not None else '사용 안 함'))
print(f'  교차 검증에서: 보류 {cv_defer:.1%}, 판정한 것의 정확도 {cv_acc:.4f}'
      f' ({"목표 달성" if cv_acc >= TARGET_ACC else "목표 미달"}),'
      f' 놓친 불량 {-best[0][0]}건 / 판정 오류 {-best[0][1]}건')

# ──────────────────────────────────────────────────────────
# [최종 학습] 기준을 정한 뒤 훈련 80% 전체로 다시 학습 (데이터를 최대한 사용)
# 표준화 기준도 훈련 80% 전체로 fit, 테스트에는 transform만
# validation_data는 에폭마다 테스트 성적을 기록만 함(그래프용) — 학습·기준 결정에는 쓰지 않음
# ──────────────────────────────────────────────────────────
sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s = sc.transform(X_test)
model = build_model()
history = model.fit(X_train_s, {'fail': y_train, 'types': T_train}, epochs=50, verbose=0,
                    validation_data=(X_test_s, {'fail': y_test, 'types': T_test}))

# ──────────────────────────────────────────────────────────
# [평가] 테스트 20% — 정해 둔 기준으로 1회만 채점
#   정확도 = (TP+TN) / 판정 수 : 판정한 것 중 맞힌 비율
#   정밀도 = TP / (TP+FP)       : 불량 판정 중 진짜 불량 (낮으면 과검)
#   재현율 = TP / (TP+FN)       : 판정한 실제 불량 중 잡아낸 비율 (낮으면 미검)
# ──────────────────────────────────────────────────────────
p_fail, p_types = model.predict(X_test_s, verbose=0)
p_fail = p_fail.ravel()

y_base = (p_fail >= 0.5).astype(int)          # 비교용: 보류 없이 0.5로만 판정
print('\n[보류 없음 — 비교용] 테스트셋')
print('정확도:', f'{accuracy_score(y_test, y_base):.4f}',
      '/ 정밀도:', f'{precision_score(y_test, y_base, zero_division=0):.4f}',
      '/ 재현율:', f'{recall_score(y_test, y_base):.4f}')

y_dec = decide(p_fail, p_types, LO, HI, TYPE_THS)
dec = y_dec != -1
tn, fp, fn, tp = confusion_matrix(y_test[dec], y_dec[dec], labels=[0, 1]).ravel()
acc = accuracy_score(y_test[dec], y_dec[dec])
print('\n[판정 보류 적용] 테스트셋')
print(f'판정 {dec.sum()}건 / 보류 {(~dec).sum()}건 ({(~dec).mean():.1%}) — 보류 중 실제 불량 {int(y_test[~dec].sum())}건')
print('1. 정확도(Accuracy) :', f'{acc:.4f}', '(판정한 것 기준)')
print('2. 정밀도(Precision):', f'{precision_score(y_test[dec], y_dec[dec], zero_division=0):.4f}')
print('3. 재현율(Recall)   :', f'{recall_score(y_test[dec], y_dec[dec], zero_division=0):.4f}')
print(f'   혼동행렬(판정한 것): 맞힌 고장 {tp} / 놓친 고장(미검) {fn} / 오탐(과검) {fp} / 맞힌 정상 {tn}')
print('   목표 정확도', TARGET_ACC, '→', '달성' if acc >= TARGET_ACC else '미달')

# 결론: 고장유형별 성적 (보류 여부와 관계없이 유형 확률 0.5 기준)
t_pred = (p_types >= 0.5).astype(int)
print('\n[고장유형별 결론] 테스트셋')
for i, name in enumerate(type_names):
    n_true = int(T_test[:, i].sum())
    n_hit = int(((T_test[:, i] == 1) & (t_pred[:, i] == 1)).sum())
    n_false = int(((T_test[:, i] == 0) & (t_pred[:, i] == 1)).sum())
    print(f'  {name}: 실제 {n_true:>2}건 중 {n_hit:>2}건 맞힘, 오탐 {n_false}건')

# Keras 내장 평가 — loss(전체), fail_loss·types_loss(출력별), fail_accuracy(보류 없이 0.5 기준 정확도)
print('\n훈련 세트 평가:', model.evaluate(X_train_s, {'fail': y_train, 'types': T_train}, verbose=0, return_dict=True))
print('테스트 세트 평가:', model.evaluate(X_test_s, {'fail': y_test, 'types': T_test}, verbose=0, return_dict=True))

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
# 샘플 1개도 2차원 (1, 6) 형태, 학습 때의 encoder·scaler·보류 기준을 그대로 사용(새로 fit 금지)
# ──────────────────────────────────────────────────────────
X_new = np.array([[
    encoder.transform(['M'])[0],        # 'M' → 2
    298.1, 308.6, 1551, 42.8, 0
]])
X_new = sc.transform(X_new)

pred_prob, type_prob = model.predict(X_new, verbose=0)
pred = decide(pred_prob.ravel(), type_prob, LO, HI, TYPE_THS)[0]
print('\n예측 확률 [고장]:', pred_prob)
print('고장유형별 확률', type_names, ':', type_prob.round(4))
print('예측 고장유형:', [n for n, v in zip(type_names, type_prob.ravel()) if v >= 0.5] or '없음')
print('예측 결과:', '판정 보류 (사람 확인)' if pred == -1 else target_names[pred])
