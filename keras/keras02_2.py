import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([2,4,6,8,10,12,14,16,18,20])
x_test = np.array(list(range(101,111)))
y_test = np.array(list(range(111,121)))
x_predict = np.array([111,112,113])

model = Sequential()
model.add(Dense(1,input_dim = 1))
#model.add(Dense(1))
model.compile(loss='mse',optimizer='adam',)
model.fit(x_train,y_train,epochs=1,batch_size=1)

loss = model.evaluate(x_test,y_test,batch_size=1)
print('loss : ',loss)

result = model.predict(x_predict)
print(result)
