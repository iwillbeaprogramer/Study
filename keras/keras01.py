import numpy as np
import tensorflow as tf

"""
1.데이터 준비
2.모델 구성
3.컴파일
4.핏

"""


x = np.array([1,2,3,4,5,6,7,8,9,10,]).reshape(-1,1)
y = np.array([2,4,6,8,10,12,14,16,18,20]).reshape(-1,1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(5,input_dim=1,activation='linear'))
model.add(Dense(3,activation='linear'))
model.add(Dense(4,activation='linear'))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
model.fit(x,y,epochs=1000,batch_size=1)

loss = model.evaluate(x,y,batch_size=1)
print('loss : ',loss)

result = model.predict([10])
print(result)
