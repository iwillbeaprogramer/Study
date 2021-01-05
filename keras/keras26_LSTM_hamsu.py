# keras 23 _LSTM3_scale을 함수형으로 코딩
import numpy as np

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70]).reshape(1,3,1)

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,LSTM,Input
from sklearn.metrics import r2_score

x = x.reshape(13,3,1)

inputs = Input(shape = (3,1))
lstm = LSTM(1024,activation='relu')(inputs)
dense1 = Dense(512,activation='relu')(lstm)
dense2 = Dense(256,activation='relu')(dense1)
dense3 = Dense(128,activation='relu')(dense2)
dense4 = Dense(64,activation='relu')(dense3)
dense5 = Dense(32,activation='relu')(dense4)
dense6 = Dense(8,activation='relu')(dense5)
dense7 = Dense(4,activation='relu')(dense6)
outputs = Dense(1)(dense7)
model = Model(inputs,outputs)

# model = Sequential()
# model.add(LSTM(1024,input_shape=(3,1),activation='relu'))
# model.add(Dense(512,activation='relu'))
# model.add(Dense(256,activation='relu'))
# model.add(Dense(128,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(32,activation='relu'))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(8,activation='relu'))
# model.add(Dense(4,activation='relu'))
# model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=500,batch_size=1)

loss = model.evaluate(x,y,batch_size=1)
y_pred = model.predict(x_pred)
print(y_pred)
print('loss : ',loss)
'''
[[81.13962]]
[[80.14889]]
loss :  0.05985087901353836
'''