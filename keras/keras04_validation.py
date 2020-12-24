import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])

x_validation = np.array([6,7,8])
y_validation = np.array([6,7,8])

x_test = np.array([9,10,11])
y_test = np.array([9,10,11])



model = Sequential()
model.add(Dense(5,input_dim = 1,activation='relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))
#model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,validation_data = (x_validation,y_validation), epochs=1000, batch_size=1)

loss = model.evaluate(x_test,y_test,batch_size=1)
print('loss : ',loss)

result = model.predict(x_train)
print(result)
