# 23-3을 카피해서 LSTM층을 두개를 만든다.

import numpy as np

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70]).reshape(1,3,1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from sklearn.metrics import r2_score

x = x.reshape(x.shape[0],x.shape[1],1)
model = Sequential()
model.add(LSTM(1024,input_shape=(3,1),activation='relu',return_sequences=True))
model.add(Dense(1024,activation='relu'))
model.add(LSTM(512,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1))
model.summary()
'''
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=500,batch_size=1)

loss = model.evaluate(x,y,batch_size=1)
y_pred = model.predict(x_pred)
print(y_pred)
print('loss : ',loss)
'''
'''
LSTM 1개
[[81.13962]]
[[80.14889]]
loss :  0.05985087901353836


LSTM 2개
[[76.38666]]
loss :  0.8654062747955322
'''

'''
input_shape가 (none,100,5) ->(none,100,10)

일단 return_sequences=True 이걸 외우자
'''