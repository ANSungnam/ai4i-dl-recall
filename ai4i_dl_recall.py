# 전처리, 표준화, 불균형 보정(재현율 개선)을 적용한 딥러닝
#
# 전체 흐름 — '전처리'와 '표준화'는 서로 다른 단계:
#   [전처리]    데이터의 내용·형태를 고침 (노이즈 행 제거, 특성 선택, 문자→숫자)
#               → 행마다 독립적으로 정해지는 작업이라 분할 '전'에 전체에 해도 안전
#   [분할]      훈련 60 / 검증 20 / 테스트 20
#   [표준화]    값의 크기(스케일)만 맞춤 (평균 0, 표준편차 1)
#               → 평균·표준편차라는 기준을 '배우는' 작업이라 분할 '후' 훈련셋으로만 fit
#   [학습+교정] class_weight 후보를 검증셋 재현율로 교정 (루프)
#   [최종 확인] 테스트셋 1회 측정 — 보고 재조정 금지

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 재현성 고정: 신경망 초기 가중치·셔플의 난수열을 시드 0으로 고정
# → 매 실행 동일한 결과. 0은 레시피 번호일 뿐, 다른 정수도 무방
tf.keras.utils.set_random_seed(0)

# 데이터 수집하기
data = pd.read_csv(
    Path('~/ws/robot/ai4i2020.csv').expanduser(),
    encoding='utf-8-sig'
)

# [전처리 1] 라벨 노이즈 제거 — 모순된 행을 지우는 '내용 정제'
# 규칙: 고장 유형(TWF~RNF) 중 하나라도 1 ↔ Machine failure = 1 이어야 함
flag_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
flag_any = data[flag_cols].any(axis=1)        # 행별로 유형 중 하나라도 1이면 True
mf = data['Machine failure'] == 1

noise = flag_any != mf                        # 플래그와 라벨이 모순인 행 (18+9=27)
print('제거할 노이즈 행 수:', noise.sum())    # 27
data = data[~noise].reset_index(drop=True)    # ~ = 반전: 모순 없는 행만 남김
print('노이즈 제거 후 데이터 수:', len(data)) # 9973

# [전처리 2] 특성 선택 — 입력으로 쓸 열만 고름
# 제외: UDI·Product ID(단순 식별자), 고장 유형 TWF~RNF(정답이 새는 누수 열)
feature_names = [
    'Type',
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

X = data[feature_names].copy()
y = data['Machine failure'].to_numpy()        # 0=정상, 1=고장
target_names = np.array(['정상', '고장'])     # 예측 번호 → 이름 변환용

# ──────────────────────────────────────────────────────────
# [학습 전 1] 라벨 분포 확인
# 고장이 약 3.3%뿐인 불균형 데이터
#  → 정확도 대신 재현율을 목표 지표로 삼고 class_weight로 보정
# ──────────────────────────────────────────────────────────
print('고장 비율:', f'{y.mean():.4f}',
      f'(정상 {np.sum(y == 0)} / 고장 {np.sum(y == 1)})')

# [전처리 3] 인코딩 — 문자를 숫자로 바꾸는 '형태 변환' (값의 의미는 그대로)
# Type: H → 0, L → 1, M → 2 (알파벳순)
encoder = LabelEncoder()
X['Type'] = encoder.fit_transform(X['Type'])
X = X.to_numpy()

# [분할] 훈련 60% / 검증 20% / 테스트 20%
#   검증셋  → 학습 감시 + COST_RATIO 교정용 (몇 번이든 참고 가능)
#   테스트셋 → 최종 확인 1회만
# 전처리(위)는 분할 전에 끝내고, 표준화(아래)는 반드시 분할 뒤에 한다
# stratify와 class_weight는 역할이 다름:
#   stratify     = 분할을 공정하게 (세 세트에 고장이 고르게 들어가 채점이 믿을 만해짐)
#   class_weight = 학습을 공정하게 (모델이 고장을 무시하지 않게)
#   → 가중치를 써도 stratify는 필요: 고장 330건뿐이라 무작위 분할이면
#     검증·테스트에 고장이 쏠리거나 모자라 재현율 측정 자체가 흔들림
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.2,           # 전체 → 훈련몫 8 : 테스트 2
    random_state=0,
    stratify=y               # 양쪽의 고장 비율을 3.3%로 동일하게 (분할 공정성)
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,          # 훈련몫 안에서 검증 생성: 0.8 × 0.25 = 전체의 20%
    random_state=0,
    stratify=y_temp          # stratify는 불균형 자체를 못 고침 → 그건 class_weight 몫
)

print('훈련:', X_train.shape, '검증:', X_val.shape, '테스트:', X_test.shape)

# [표준화] 전처리와 다른 점: 데이터를 '고치는' 게 아니라,
# 훈련셋의 평균·표준편차라는 '기준을 배워서' 값의 크기만 맞추는 단계.
# 기준을 배우기 때문에 반드시 분할 후, 훈련셋으로만 fit한다.
# (검증·테스트까지 섞어 fit하면 시험 문제를 미리 본 셈 = 데이터 누수)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # 평균·표준편차는 훈련셋에서만 계산(fit)
X_val = scaler.transform(X_val)           # 훈련 기준을 적용만 — 다시 fit하면 데이터 누수
X_test = scaler.transform(X_test)         # 동일

# ──────────────────────────────────────────────────────────
# [학습 전 2] class_weight 기본값 계산
# 역빈도 가중치(정상 수 / 고장 수)를 주면 고장의 손실 비중이
# 정확히 50%가 됨 = 고장 샘플을 복제해 50:50으로 불린 것과 동치.
#
# 단, 50%는 '최적점'이 아니라 비용 정보가 없을 때의 '출발점'.
# 최적 가중치 ∝ (정상 수 × FN비용) / (고장 수 × FP비용) 이므로,
# COST_RATIO(=FN비용/FP비용) 후보를 검증셋으로 비교해 고른다.
# ──────────────────────────────────────────────────────────
n_normal = (y_train == 0).sum()
n_fail = (y_train == 1).sum()
base_weight = n_normal / n_fail        # ≈ 29 (고장 비중 50%가 되는 값)

def build_model():                            # 루프에서 후보마다 새 모델로 시작하기 위해 함수화
    m = Sequential()
    m.add(Dense(64, activation='relu', input_dim=6))   # 입력 = 특성 6개
    m.add(Dense(64, activation='relu'))
    m.add(Dense(2, activation='softmax'))     # 출력 [정상 확률, 고장 확률], 합 = 1
    m.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',   # 정수 라벨(0/1)을 그대로 쓰는 분류 손실
        metrics=['accuracy']                  # 참고용일 뿐 — 불균형에선 판단 지표로 안 씀
    )
    return m

# ──────────────────────────────────────────────────────────
# [학습 중 + 교정] 검증셋 전체로 COST_RATIO 교정 (루프)
# 후보마다 재학습 → 검증 재현율로 채점 → 목표 달성하는
# 가장 작은 비율 채택 (작을수록 오탐이 적으므로)
# 테스트셋은 이 루프에 절대 쓰지 않는다
# ──────────────────────────────────────────────────────────
TARGET_RECALL = 0.90                   # 목표: 고장의 90% 이상 잡기

model = None
for cost_ratio in [1.0, 2.0, 3.0, 5.0]:       # 1.0 = 50:50 출발점, 키울수록 고장 쪽 편향
    class_weight = {0: 1.0, 1: cost_ratio * base_weight}
    m = build_model()
    hist = m.fit(
        X_train, y_train,
        epochs=50,
        validation_data=(X_val, y_val),       # 에폭마다 검증 성적 기록 → 과적합 감시
        class_weight=class_weight,            # 고장 손실 ×가중치 = 복제 오버샘플링과 동치
        verbose=0
    )
    val_pred = np.argmax(m.predict(X_val, verbose=0), axis=1)   # 확률 큰 쪽(0.5 기준) 판정
    val_recall = recall_score(y_val, val_pred)
    val_precision = precision_score(y_val, val_pred)
    print(f'COST_RATIO {cost_ratio:.1f} (가중치 1:{cost_ratio*base_weight:.0f})'
          f' → 검증 재현율 {val_recall:.3f}, 검증 정밀도 {val_precision:.3f}')
    if model is None or val_recall >= TARGET_RECALL:
        model, history = m, hist       # 일단 채택
    if val_recall >= TARGET_RECALL:
        print(f'  → 목표({TARGET_RECALL}) 달성, COST_RATIO {cost_ratio} 확정')
        break
else:
    print(f'  → 목표 미달: 마지막 모델 채택 (데이터 보강 필요 신호)')

# ──────────────────────────────────────────────────────────
# [학습 후] 최종 확인 — 교정이 끝난 모델을 테스트셋으로 1회만 측정
# 재현율 = TP / (TP + FN) : 실제 고장 중 잡아낸 비율
# 여기 결과를 보고 다시 조정하면 안 됨 (조정은 검증셋의 몫)
# ──────────────────────────────────────────────────────────
test_prob = model.predict(X_test)
y_pred = np.argmax(test_prob, axis=1)         # 고장 확률 ≥ 0.5 → 고장(1)

cm = confusion_matrix(y_test, y_pred)
print('\n=== 테스트셋 평가 ===')
print('혼동행렬 [[TN FP], [FN TP]]:')
print(cm)
print('놓친 고장(FN):', cm[1, 0], '건 / 오탐(FP):', cm[0, 1], '건')   # FN이 가장 비싼 실수
print('재현율(Recall) :', f'{recall_score(y_test, y_pred):.3f}')      # TP/(TP+FN): 실제 고장 중 잡은 비율
print('정밀도(Precision):', f'{precision_score(y_test, y_pred):.3f}') # TP/(TP+FP): 고장 예측 중 진짜 비율
print('F1 점수        :', f'{f1_score(y_test, y_pred):.3f}')          # 재현율·정밀도의 조화평균
# 판단 기준: 재현율만 보면 안 됨 ('전부 고장' 예측도 재현율 100%)
#  → 주 지표 = 재현율, 하한선 = 정밀도(감당 가능한 오탐 수)로 함께 판단
# 검증 재현율보다 테스트 재현율이 조금 낮게 나오는 건 정상
# (검증셋에 맞춰 고른 만큼의 차이 — 테스트 숫자가 진짜 실력)

# 학습 과정 그래프 (검증셋 기준)
epochs = np.arange(1, len(history.history['loss']) + 1)

plt.figure(figsize=(7, 4))
plt.plot(epochs, history.history['loss'], label='Training loss')
plt.plot(epochs, history.history['val_loss'], label='Validation loss')
plt.plot(epochs, history.history['accuracy'], label='Training acc')
plt.plot(epochs, history.history['val_accuracy'], label='Validation acc')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# 새로운 데이터 예측하기
# 순서: Type, 공기 온도, 공정 온도, 회전 속도, 토크, 공구 마모
X_new = np.array([[
    encoder.transform(['M'])[0],              # 학습 때의 encoder 재사용: 'M' → 2
    298.1, 308.6, 1551, 42.8, 0
]])                                           # 2차원 (1, 6) 형태 — 샘플 1개도 배치 차원 필요

X_new = scaler.transform(X_new)               # 학습 때 fit한 scaler 재사용 (새로 fit 금지)

# 정상과 고장의 예측 확률
pred_prob = model.predict(X_new)
print('예측 확률 [정상, 고장]:', pred_prob)

pred = np.argmax(pred_prob, axis=1)           # 확률 큰 쪽 선택 (0.5 기준)
print('예측 번호:', pred)
print('예측 결과:', target_names[pred])       # 0→정상, 1→고장
