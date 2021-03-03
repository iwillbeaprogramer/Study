import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import mnist
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(-1,28,28,1)/255.


modelpath = '../data/modelcheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor='val_loss',patience=10)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

with strategy.scope():
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
    model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

hist = model.fit(x_train,y_train,validation_split = 0.2,epochs=30,verbose=1,batch_size=16,callbacks=[es,cp])

loss = model.evaluate(x_test,y_test,batch_size=16)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict,axis=-1)
