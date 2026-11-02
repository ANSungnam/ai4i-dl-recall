# 가중치 x 판정 임계값 실험 (원본 6특성 vs 파생 9특성)
# 임계값은 검증셋에서만 고름: 검증 재현율 >= 0.80 을 지키는 가장 높은 임계값
# 테스트셋은 고른 뒤 1회 측정
from pathlib import Path
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import recall_score, precision_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

MIN_VAL_RECALL = 0.80
data = pd.read_csv(Path('~/ws/robot/ai4i2020.csv').expanduser(), encoding='utf-8-sig')
flags = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
data = data[data[flags].any(axis=1) == (data['Machine failure'] == 1)].reset_index(drop=True)
base = ['Type', 'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
X = data[base].copy()
X['Type'] = LabelEncoder().fit_transform(X['Type'])
X['dT'] = X['Process temperature [K]'] - X['Air temperature [K]']
X['Power'] = X['Torque [Nm]'] * X['Rotational speed [rpm]'] * 2 * np.pi / 60
X['Strain'] = X['Tool wear [min]'] * X['Torque [Nm]']
y = data['Machine failure'].to_numpy()
idx = np.arange(len(y))
i_tmp, i_te = train_test_split(idx, test_size=0.2, random_state=0, stratify=y)
i_tr, i_va = train_test_split(i_tmp, test_size=0.25, random_state=0, stratify=y[i_tmp])
ratio = (y[i_tr] == 0).sum() / (y[i_tr] == 1).sum()

print(f'임계값 선택 규칙: 검증 재현율 >= {MIN_VAL_RECALL} 중 가장 높은 임계값')
print('특성 | 가중치 | 임계값 | 검증 R/P | 테스트 재현율 | 테스트 정밀도 | FN | FP')
for name, cols in [('원본6', base), ('파생9', list(X.columns))]:
    A = X[cols].to_numpy(dtype=float)
    sc = StandardScaler().fit(A[i_tr])
    Atr, Ava, Ate = sc.transform(A[i_tr]), sc.transform(A[i_va]), sc.transform(A[i_te])
    for w in [1.0, 3.0, 10.0, ratio]:
        tf.keras.utils.set_random_seed(0)
        m = Sequential([Input((Atr.shape[1],)), Dense(64, activation='relu'),
                        Dense(64, activation='relu'), Dense(2, activation='softmax')])
        m.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        m.fit(Atr, y[i_tr], epochs=50, class_weight={0: 1.0, 1: w}, verbose=0)
        pva = m.predict(Ava, verbose=0)[:, 1]
        pte = m.predict(Ate, verbose=0)[:, 1]
        best = None
        for t in np.linspace(0.01, 0.99, 99):
            if recall_score(y[i_va], pva >= t) >= MIN_VAL_RECALL:
                best = t
        if best is None:
            best = 0.01
        vr = recall_score(y[i_va], pva >= best); vp = precision_score(y[i_va], pva >= best, zero_division=0)
        yp = pte >= best
        cm = confusion_matrix(y[i_te], yp)
        print(f'{name} | {w:5.1f} | {best:.2f} | {vr:.3f}/{vp:.3f} | '
              f'{recall_score(y[i_te], yp):.3f} | {precision_score(y[i_te], yp, zero_division=0):.3f} | '
              f'{cm[1,0]} | {cm[0,1]}', flush=True)
