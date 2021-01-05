# 23 LSTM3_scale 을 dnn코딩해서 비교
# DNN으로 23번 파일보다 loss를 좋게 만들것
import numpy as np

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([[50,60,70]])
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,SimpleRNN,GRU
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)
x_pred = scaler.transform(x_pred)
model = Sequential()
model.add(Dense(1024,activation='relu',input_dim=3))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=1000,batch_size=4)

loss = model.evaluate(x,y,batch_size=4)
y_pred = model.predict(x_pred)
print(y_pred)
print('loss : ',loss)
'''
LSTM
[[80.14889]]
loss :  0.05985087901353836

SimpleRNN
[[83.46045]]
loss :  0.55766761302948

GRU
[[80.35569]]
loss :  0.0010759884025901556

DNN
loss :  0.0003774349170271307

'''