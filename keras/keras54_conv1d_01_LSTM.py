#23 - 3을 카피해서 conv1d로 완성할것
# LSTM과비교
import numpy as np

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70]).reshape(1,3,1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv1D,Flatten,MaxPooling1D,Dropout
from sklearn.metrics import r2_score

x = x.reshape(13,3,1)
model = Sequential()
model.add(Conv1D(filters = 1024,kernel_size=2,input_shape=(3,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1))
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

conv1d
[[79.89358]]
loss :  2.4019200801849365
'''