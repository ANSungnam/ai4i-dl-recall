# 실험2: 물리 경계로부터의 거리 특성 + 긴 학습/조기종료 + 시드 5개 평균
# 임계값은 검증셋에서만 선택(검증 재현율 >= 0.80 중 최고 임계값), 테스트는 1회 측정
from pathlib import Path
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

MIN_VAL_RECALL = 0.80
d = pd.read_csv(Path('~/ws/robot/ai4i2020.csv').expanduser(), encoding='utf-8-sig')
flags = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
d = d[d[flags].any(axis=1) == (d['Machine failure'] == 1)].reset_index(drop=True)

air, proc = d['Air temperature [K]'], d['Process temperature [K]']
rpm, tq, wear = d['Rotational speed [rpm]'], d['Torque [Nm]'], d['Tool wear [min]']
type_num = d['Type'].map({'L': 0, 'M': 1, 'H': 2})
base = pd.DataFrame({'Type': type_num, 'Air': air, 'Proc': proc, 'Rpm': rpm, 'Torque': tq, 'Wear': wear})

P = tq * rpm * 2 * np.pi / 60
limit = d['Type'].map({'L': 11000, 'M': 12000, 'H': 13000})
margin = base.copy()
margin['HDF_dT'] = (8.6 - (proc - air))              # >0 이면 방열 조건 쪽
margin['HDF_rpm'] = (1380 - rpm) / 100
margin['HDF_both'] = np.minimum(margin['HDF_dT'], margin['HDF_rpm'])
margin['PWF_low'] = (3500 - P) / 1000
margin['PWF_high'] = (P - 9000) / 1000
margin['PWF_out'] = np.maximum(margin['PWF_low'], margin['PWF_high'])
margin['OSF'] = (wear * tq - limit) / 1000
margin['TWF'] = (wear - 200) / 10

y = d['Machine failure'].to_numpy()
idx = np.arange(len(y))
i_tmp, i_te = train_test_split(idx, test_size=0.2, random_state=0, stratify=y)
i_tr, i_va = train_test_split(i_tmp, test_size=0.25, random_state=0, stratify=y[i_tmp])
# 원래 못 맞히는 고장(TWF 단독, RNF 단독) 표시 — 참고용 정밀도/재현율 계산에만 사용
unpred = ((d['TWF'] == 1) | (d['RNF'] == 1)) & (d[['HDF', 'PWF', 'OSF']].sum(axis=1) == 0)

print(f'임계값 규칙: 검증 재현율 >= {MIN_VAL_RECALL} 중 최고 임계값 / 시드 5개')
print('특성 | 가중치 | 시드 | 테스트 재현율 | 테스트 정밀도 | FN | FP | 예측가능고장 재현율')
for name, X in [('원본6', base), ('경계거리', margin)]:
    A = X.to_numpy(dtype=float)
    sc = StandardScaler().fit(A[i_tr])
    Atr, Ava, Ate = sc.transform(A[i_tr]), sc.transform(A[i_va]), sc.transform(A[i_te])
    for w in [1.0, 3.0]:
        rs, ps = [], []
        for seed in range(5):
            tf.keras.utils.set_random_seed(seed)
            m = Sequential([Input((Atr.shape[1],)), Dense(64, activation='relu'),
                            Dense(64, activation='relu'), Dense(2, activation='softmax')])
            m.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            m.fit(Atr, y[i_tr], epochs=300, batch_size=64, class_weight={0: 1.0, 1: w},
                  validation_data=(Ava, y[i_va]), verbose=0,
                  callbacks=[EarlyStopping(patience=30, restore_best_weights=True)])
            pva = m.predict(Ava, verbose=0)[:, 1]; pte = m.predict(Ate, verbose=0)[:, 1]
            best = 0.01
            for t in np.linspace(0.01, 0.99, 99):
                if recall_score(y[i_va], pva >= t) >= MIN_VAL_RECALL:
                    best = t
            yp = pte >= best
            cm = confusion_matrix(y[i_te], yp)
            r = recall_score(y[i_te], yp); p = precision_score(y[i_te], yp, zero_division=0)
            pred_mask = ~unpred.to_numpy()[i_te]
            r2 = recall_score(y[i_te][pred_mask], yp[pred_mask])
            rs.append(r); ps.append(p)
            print(f'{name} | {w:.0f} | {seed} | {r:.3f} | {p:.3f} | {cm[1,0]} | {cm[0,1]} | {r2:.3f}', flush=True)
        print(f'  >> {name} 가중치 {w:.0f} 평균: 재현율 {np.mean(rs):.3f}, 정밀도 {np.mean(ps):.3f}', flush=True)
