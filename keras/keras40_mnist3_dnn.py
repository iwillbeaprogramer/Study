# input_shape = (28*28,)

# 인공지능계의 hello world mnist

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(60000,28*28).astype('float32')/255.
x_test = x_test.reshape(-1,28*28)/255.
# (x_test.reshape(x_test[0],x_test.shape[1],x_test.shape[2],1))

from tensorflow.keras.utils import to_categorical



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='loss',patience=10)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Dense(1024,activation='relu',input_dim=28*28))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,validation_split = 0.2,epochs=300,verbose=1,batch_size=32,callbacks=[es])

loss = model.evaluate(x_test,y_test,batch_size=32)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict,axis=-1)

for i in range(10):
    print('실제 : ',np.argmax(y_test[i]),'예상 : ',y_predict[i])
print('accuracy : ',loss[1])


#0.985 이상
'''
313/313 [==============================] - 2s 5ms/step - loss: 0.0931 - accuracy: 0.9874
실제 :  7 예상 :  7
실제 :  2 예상 :  2
실제 :  1 예상 :  1
실제 :  0 예상 :  0
실제 :  4 예상 :  4
실제 :  1 예상 :  1
실제 :  4 예상 :  4
실제 :  9 예상 :  9
실제 :  5 예상 :  5
실제 :  9 예상 :  9
accuracy :  0.9873999953269958
'''

