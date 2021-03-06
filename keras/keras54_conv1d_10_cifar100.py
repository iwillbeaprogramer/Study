import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar100
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,LSTM,Conv1D,Flatten,MaxPooling1D,Dropout
from tensorflow.keras.callbacks import EarlyStopping


es = EarlyStopping(monitor='loss',patience = 10)
(x_train,y_train),(x_test,y_test) = cifar100.load_data()
#  (50000, 32, 32, 3) (50000, 1)
#  (10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(-1,32,32*3).astype('float32')/255
x_test= x_test.reshape(-1,32,32*3).astype('float32')/255

onehot = OneHotEncoder()
y_train = onehot.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = onehot.transform(y_test.reshape(-1,1)).toarray()

print(y_test.shape)
model = Sequential()
model.add(Conv1D(filters = 256,kernel_size=2,input_shape=(32,32*3)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(100,activation='softmax'))
model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics = ['accuracy'])
model.fit(x_train,y_train,validation_split=0.2,epochs=200,callbacks=[es],batch_size=16,verbose=1)

loss = model.evaluate(x_test,y_test,batch_size=16)
y_predict = model.predict(x_test)
y_test = np.argmax(y_test,axis=-1)
y_predict = np.argmax(y_predict,axis=-1)

for i in range(10,20):
    print('예상 : {}    실제 : {}'.format(y_predict[i],y_test[i]))

print('loss : {}    accuracy : {}'.format(loss[0],loss[1]))


'''
CNN
loss : 3.8951032161712646    accuracy : 0.24940000474452972
DNN
accuracy : 0.22
LSTM


conv1d
loss : 5.286771297454834    accuracy : 0.21629999577999115
'''