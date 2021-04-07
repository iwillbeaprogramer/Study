import numpy as np
from tensorflow.keras.datasets import mnist

(x_train,_),(x_test,_) = mnist.load_data()

x_train = x_train.reshape(60000,784).astype('float32')/255.0
x_test = x_test.reshape(10000,784).astype('float32')/255.0

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size,input_shape=(784,),activation='relu'))
    model.add(Dense(units=784,activation='sigmoid'))
    return model


model_01 = autoencoder(hidden_layer_size=1)
model_02 = autoencoder(hidden_layer_size=2)
model_04 = autoencoder(hidden_layer_size=4)
model_08 = autoencoder(hidden_layer_size=8)
model_16 = autoencoder(hidden_layer_size=16)
model_32 = autoencoder(hidden_layer_size=32)

print("node 1개 시작")
model_01.compile(optimizer = 'adam',loss = 'binary_crossentropy')
model_01.fit(x_train,x_train,epochs=10)

print("node 2개 시작")
model_02.compile(optimizer = 'adam',loss = 'binary_crossentropy')
model_02.fit(x_train,x_train,epochs=10)

print("node 4개 시작")
model_04.compile(optimizer = 'adam',loss = 'binary_crossentropy')
model_04.fit(x_train,x_train,epochs=10)

print("node 8개 시작")
model_08.compile(optimizer = 'adam',loss = 'binary_crossentropy')
model_08.fit(x_train,x_train,epochs=10)

print("node 16개 시작")
model_16.compile(optimizer = 'adam',loss = 'binary_crossentropy')
model_16.fit(x_train,x_train,epochs=10)

print("node 32개 시작")
model_32.compile(optimizer = 'adam',loss = 'binary_crossentropy')
model_32.fit(x_train,x_train,epochs=10)

output_01 = model_01.predict(x_test)
output_02 = model_02.predict(x_test)
output_04 = model_04.predict(x_test)
output_08 = model_08.predict(x_test)
output_16 = model_16.predict(x_test)
output_32 = model_32.predict(x_test)

import matplotlib.pyplot as plt
import random

fig,axes = plt.subplots(7,5,figsize=(15,15))
random_imgs = random.sample(range(output_01.shape[0]),5)
outputs = [x_test,output_01,output_02,output_04,output_08,output_16,output_32]

for row_num,row in enumerate(axes):
    for col_num,ax in enumerate(row):
        ax.imshow(outputs[row_num][random_imgs[col_num]].reshape(28,28),cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.show()
