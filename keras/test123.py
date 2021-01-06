###행이 다른 앙상블

import numpy as np
from numpy import array

#1. 데이터
x1 = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,0], [9,10,11], [10,11,12]])
x2 = np.array([[10,20,30], [20,30,40], [30,40,50], [40,50,60], [50,60,70], 
          [60,70,80], [70,80,90], [80,90,100], [90,100,110], [100,110,120], 
        [2,3,4], [3,4,5], [4,5,6]])
y1 = array([4,5,6,7,8,9,10,11,12,13])
y2 = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x1_predict = array([55,65,75])
x2_predict = array([65,75,85])

x1 = x1.reshape(x1.shape[0], x1.shape[1], 1) #(13, 3, 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1) #(13, 3, 1)
x1_predict = x1_predict.reshape(1, 3, 1) #(1, 3, 1)
x2_predict = x2_predict.reshape(1, 3, 1) #(1, 3, 1)

#2. 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate

input1 = Input(shape=(3,1))
lstm1 = LSTM(27, activation='relu')(input1)
dense1 = Dense(18, activation='relu')(lstm1)
dense1 = Dense(9, activation='relu')(dense1)

input2 = Input(shape=(3,1))
lstm2 = LSTM(27, activation='relu')(input1)
dense2 = Dense(18, activation='relu')(lstm1)
dense2 = Dense(9, activation='relu')(dense1)

merge1 = concatenate([dense1, dense2])
middle = Dense(18)(merge1)
middle = Dense(18)(middle)

output1 = Dense(36)(middle)
output1 = Dense(18)(middle)
output1 = Dense(9)(output1)
output1 = Dense(1)(output1)

output2 = Dense(36)(middle)
output2 = Dense(18)(output2)
output2 = Dense(9)(output2)
output2 = Dense(1)(output2)

model = Model(inputs = [input1, input2], outputs = [output1,output2])

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit([x1, x2], [y1, y2], epochs=200, verbose=1)

#평가, 예측
loss = model.evaluate([x1, x2], [y1,y2])
print('loss: ', loss)

y_pred= model.predict([x1_predict, x2_predict])
print('y_pred: ', y_pred)