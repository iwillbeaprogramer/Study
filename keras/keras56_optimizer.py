import numpy as np
# 데이터

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2 모델구성
model = Sequential()
model.add(Dense(1000,input_dim = 1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3컴파일 훈련
from tensorflow.keras.optimizers import Adam,Adadelta,Adamax,Adagrad
from tensorflow.keras.optimizers import RMSprop,SGD,Nadam

#optimizer = Adam(lr=0.01)
# adam
#0.001        10/10 [==============================] - 0s 1ms/step - loss: 9.2954e-10 - mse: 9.2954e-10
#0.1          10/10 [==============================] - 0s 1ms/step - loss: 3.6821e-06 - mse: 3.6821e-06
#0.01         10/10 [==============================] - 0s 1ms/step - loss: 2.3874e-13 - mse: 2.3874e-13

optimizer = Nadam(lr=0.1)
#adadelta
#0.1           10/10 [==============================] - 0s 997us/step - loss: 1.3457e-05 - mse: 1.3457e-05
#0.01          10/10 [==============================] - 0s 1ms/step - loss: 1.0790e-05 - mse: 1.0790e-05
#0.001         10/10 [==============================] - 0s 3ms/step - loss: 1.0077 - mse: 1.0077

# Adamax
#0.1            10/10 [==============================] - 0s 1ms/step - loss: 2.0314e-05 - mse: 2.0314e-05
#0.01           10/10 [==============================] - 0s 1ms/step - loss: 4.7891e-13 - mse: 4.7891e-13
#0.001          10/10 [==============================] - 0s 1ms/step - loss: 4.1631e-09 - mse: 4.1631e-09

# Adagrad
#0.1            10/10 [==============================] - 0s 1ms/step - loss: 96.7470 - mse: 96.7470
#0.01           10/10 [==============================] - 0s 2ms/step - loss: 5.1909e-08 - mse: 5.1909e-08
#0.001          10/10 [==============================] - 0s 1ms/step - loss: 3.9523e-05 - mse: 3.9523e-05

# SGD
#0.1            10/10 [==============================] - 0s 1ms/step - loss: 96.7470 - mse: 96.7470
#0.01           10/10 [==============================] - 0s 2ms/step - loss: nan - mse: nan
#0.001          10/10 [==============================] - 0s 3ms/step - loss: nan - mse: nan

# RMSprop
#0.1            10/10 [==============================] - 0s 3ms/step - loss: 1714719424512.0000 - mse: 1714719424512.0000
#0.01           10/10 [==============================] - 0s 898us/step - loss: 25.1041 - mse: 25.1041
#0.001          10/10 [==============================] - 0s 1ms/step - loss: 0.5086 - mse: 0.5086

# Nadam
#0.1            10/10 [==============================] - 0s 4ms/step - loss: 7.8650e-04 - mse: 7.8650e-04
#0.01           10/10 [==============================] - 0s 5ms/step - loss: 137374.3438 - mse: 137374.3438
#0.001          10/10 [==============================] - 0s 1ms/step - loss: 5.1477e-05 - mse: 5.1477e-05
model.compile(loss = 'mse',optimizer = optimizer,metrics = ['mse'])
model.fit(x,y,epochs = 100,batch_size = 1)

loss,mse = model.evaluate(x,y,batch_size=1)
y_pred = model.predict([11])
print('loss : ',loss , "결과물 : ",y_pred)