# keras23_LSTM3_scale 을 DNN으로 코딩
# 결과치 비교

# DNN으로 23번 파일보다 loss를 좋게 만들 것

import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70]) 

x_pred = np.array([50,60,70])

print(x.shape)      # (13, 3)
print(y.shape)      # (13,)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(3,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1, batch_size=1, verbose=2)

loss = model.evaluate(x, y)
print('loss :', loss)
y_pred = model.predict(np.array([x[0]]))
print('y_pred :', y_pred)

# loss : 0.0026478709187358618
# y_pred : [[80.69948]]