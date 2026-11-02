# 원본 코드에 물리 파생변수 3개를 추가한 비교용 스크립트를 생성
# dT = 공정온도 - 공기온도 (HDF), Power = 토크 x 각속도 [W] (PWF), Strain = 공구마모 x 토크 (OSF)
src = open('ai4i_dl_recall.py', encoding='utf-8').read()
src = src.split('# 새로운 데이터 예측하기')[0]  # 6특성 고정 예측부는 제외
src = src.replace(
    "X = data[feature_names].copy()\n",
    "X = data[feature_names].copy()\n"
    "X['dT'] = X['Process temperature [K]'] - X['Air temperature [K]']\n"
    "X['Power'] = X['Torque [Nm]'] * X['Rotational speed [rpm]'] * 2 * np.pi / 60\n"
    "X['Strain'] = X['Tool wear [min]'] * X['Torque [Nm]']\n",
)
src = src.replace('input_dim=6', 'input_dim=X_train.shape[1]')
assert "X['dT']" in src and 'input_dim=X_train.shape[1]' in src
open('ai4i_dl_features.py', 'w', encoding='utf-8').write(src)
print('ai4i_dl_features.py 생성 완료')
