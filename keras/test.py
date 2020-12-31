# 실습 : 19_1, 2, 3, 4, 5, EarlyStopping까지
# 총 6개의 파일을 완성

import numpy as np
from sklearn.datasets import load_diabetes

#1 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x[:5])
print(y[:10])
print(x.shape)
print(y.shape)

print(np.max(x), np.min(y))
print(dataset.feature_names)
print(dataset.DESCR)

# x = x/442

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state= 104, shuffle=True)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state= 104, shuffle=True)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#2 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(120, input_dim=10, activation='relu'))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(1))

#3 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=1000, batch_size=6, validation_data=(x_val,y_val), verbose=1)

#4 평가 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=6)
print('loss,mae : ', loss, mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(y_test, y_predict ))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2: ', r2)

# 기본
# loss,mae :  1038.3603515625 18.367414474487305
# RMSE :  32.223597019036255
# R2:  0.8205333476009842

# x통짜
# loss,mae :  2768.078125 42.41880416870117
# RMSE :  52.612524966998045
# R2:  0.5215748340033284

# 민 맥스 스케일러 x 전체를 다
# loss,mae :  817.8695068359375 11.969538688659668
# RMSE :  28.598421271010697
# R2:  0.8586421779955937

# x_train 
# loss,mae :  802.31591796875 14.623322486877441
# RMSE :  28.325182180985834
# R2:  0.861330435961635

# x_val
# loss,mae :  36.10376739501953 5.0196919441223145
# RMSE :  6.008641561271572
# R2:  0.9937599463818423