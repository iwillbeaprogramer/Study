# 열이 달라야함
import numpy as np

x1=np.array([[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[20,30],[30,40],[40,50]]).reshape(13,2,1)
x2=np.array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],[50,60,70],[60,70,80],[70,80,90],[80,90,10],[90,100,110],[100,110,120],[2,3,4],[3,4,5],[4,5,6]]).reshape(13,3,1)
y1=np.array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],[50,60,70],[60,70,80],[70,80,90],[80,90,10],[90,100,110],[100,110,120],[2,3,4],[3,4,5],[4,5,6]]).reshape(13,3)
y2=np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x1_predict = np.array([55,65]).reshape(1,2,1)
x2_predict = np.array([65,75,85]).reshape(1,3,1)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM,Dense,Input,concatenate

inputs1 = Input(shape=(2,1))
lstm1 = LSTM(16,activation='relu')(inputs1)
dense1 = Dense(16,activation='relu')(lstm1)
dense1 = Dense(8,activation='relu')(dense1)
inputs2 = Input(shape=(3,1))
lstm2 = LSTM(32,activation='relu',return_sequences=True)(inputs2)
lstm2 = LSTM(16,activation='relu')(lstm2)
dense2 = Dense(16,activation='relu')(lstm2)
dense2 = Dense(8,activation='relu')(dense2)

concat = concatenate([dense1,dense2])
dense = Dense(4)(concat)
outputs1 = Dense(3)(dense)
outputs2 = Dense(1)(dense)

model = Model([inputs1,inputs2],[outputs1,outputs2])
model.compile(loss='mse',optimizer='adam')
model.fit([x1,x2],[y1,y2],epochs=1000,batch_size=1)

loss = model.evaluate([x1,x2],[y1,y2],batch_size=1)
y1_pred = model.predict([x1_predict,x2_predict])
print(y1_pred)
print('loss : ',loss)

# predict 는 85근사치
'''
[[85.69777]]
loss :  0.007422684691846371
'''