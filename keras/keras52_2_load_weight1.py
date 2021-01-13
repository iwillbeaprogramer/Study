import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(-1,28,28,1)/255.

modelpath = '../data/modelcheckpoint/k52_1_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
#cp = ModelCheckpoint(filepath = modelpath, monitor='val_loss',save_best_only = True, mode = 'auto')
#es = EarlyStopping(monitor='val_loss',patience=10)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(filters = 512,kernel_size = (2,2),padding='same',strides = 1,input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))
model.add(Conv2D(256,kernel_size=(2,2)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(10,activation='softmax'))
#model.save('../data/h5/k_52_1_save_model1.h5')
model.save_weights('../data/h5/k_52_1_weights.h5')
model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.load_weights('../data/h5/k52_1_weight.h5')
loss = model.evaluate(x_test,y_test)
print('loss : ',loss[0])
print('acc : ',loss[1])


'''
loss :  0.08076740801334381
acc :  0.9868999719619751
'''

model2 = load_model('../data/h5/k_52_1_save_model2.h5')
loss = model2.evaluate(x_test,y_test)
print('loss : ',loss[0])
print('acc : ',loss[1])