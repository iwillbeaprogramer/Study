import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,LSTM,Dropout
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

modelpath = '../data/modelCheckpoint/k46_cifar10_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(monitor = 'val_loss',filepath=modelpath,save_best_only=True)
es = EarlyStopping(monitor='val_loss',patience = 10)
x_train = np.load('../data/npy/cifar10_x_train.npy') 
x_test = np.load('../data/npy/cifar10_x_test.npy') 
y_train = np.load('../data/npy/cifar10_y_train.npy') 
y_test = np.load('../data/npy/cifar10_y_test.npy') 

x_train = x_train.reshape(-1,32,32,3).astype('float32')/255
x_test= x_test.reshape(-1,32,32,3).astype('float32')/255

onehot = OneHotEncoder()
y_train = onehot.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = onehot.transform(y_test.reshape(-1,1)).toarray()

print(y_test.shape)
model = Sequential()
model.add(Conv2D(filters = 1024,kernel_size = 2, strides=1 , input_shape=(32,32,3),padding='valid'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.4))
model.add(Conv2D(filters = 512,kernel_size = 2, strides=1 , input_shape=(32,32,3),padding='valid'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics = ['accuracy'])
model.fit(x_train,y_train,validation_split=0.2,epochs=200,callbacks=[es,cp],batch_size=16,verbose=1)

loss = model.evaluate(x_test,y_test,batch_size=16)
y_predict = model.predict(x_test)
y_test = np.argmax(y_test,axis=-1)
y_predict = np.argmax(y_predict,axis=-1)

for i in range(10,20):
    print('예상 : {}    실제 : {}'.format(y_predict[i],y_test[i]))

print('loss : {}    accuracy : {}'.format(loss[0],loss[1]))


'''
CNN
loss : 1.019891619682312    accuracy : 0.6223
modelcheckpoint : loss : 1.2806475162506104    accuracy : 0.6466000080108643
DNN
loss : 3.381682872772217    accuracy : 0.44920000433921814

LSTM
loss : 2.302645444869995    accuracy : 0.10000000149011612
'''