import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

x_train = np.array(list(range(1,11)))
y_train = np.array(list(range(1,11)))

x_test = np.array(list(range(11,16)))
y_test = np.array(list(range(11,16)))

x_pred = np.array(list(range(16,19)))

model = Sequential()
model.add(Dense(10,input_dim = 1,activation='relu'))
model.add(Dense(5))
model.add(Dense(1))

model.compile(loss = 'mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs = 1000,batch_size=1, validation_split=0.2)

results = model.evaluate(x_test,y_test,batch_size=1)
print(results)
y_pred = model.predict(x_pred)
print("y_predict : ",y_pred)
