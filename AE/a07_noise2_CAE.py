import numpy as np
from tensorflow.keras.datasets import mnist

(x_train,_),(x_test,_) = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.0
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.0


x_train_noised = x_train + np.random.normal(0,0.1,size=x_train.shape)
x_test_noised = x_test + np.random.normal(0,0.1,size=x_test.shape)
x_train_noised = np.clip(x_train_noised,a_min = 0,a_max=1)
x_test_noised = np.clip(x_test_noised,a_min=0,a_max=1)

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,MaxPooling2D,BatchNormalization,Dropout,LeakyReLU,Input,ReLU

def autoencoder(hidden_layer_size):
    inputs = Input(shape = (28,28,1))
    layer1 = Conv2D(filters=64,kernel_size=2,padding='same',use_bias=False,strides=2)(inputs)
    layer1 = LeakyReLU()(layer1)
    layer1_ = layer1

    layer2 = Conv2D(filters=128,kernel_size=2,strides=2,padding='same',use_bias=False)(layer1)
    layer2_ = BatchNormalization()(layer2)
    layer2 = LeakyReLU()(layer2_)

    layer3 = Conv2DTranspose(filters=64,kernel_size=2,strides=2,padding='same',use_bias=False)(layer2)
    layer3 = BatchNormalization()(layer3)
    layer3 = layer3 + layer1_
    layer3 = Dropout(0.25)(layer3)
    layer3 = ReLU()(layer3)

    layer4 = Conv2DTranspose(filters=1,kernel_size=2,strides=2,padding='same',use_bias=False)(layer3)
    layer4 = BatchNormalization()(layer4)
    layer4 = layer4
    layer4 = Dropout(0.25)(layer4)
    layer4 = ReLU()(layer4)
    outputs = layer4
    model = Model(inputs=inputs,outputs=outputs)
    return model

model = autoencoder(hidden_layer_size=154)
model.summary()

model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics=['acc'])
model.fit(x_train_noised,x_train,epochs=100)
output = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random
fig,((ax1,ax2,ax3,ax4,ax5),(ax6,ax7,ax8,ax9,ax10),(ax11,ax12,ax13,ax14,ax15)) = plt.subplots(3,5,figsize=(20,7))

# 이미지 5개를 무작위로 고른다

random_images = random.sample(range(output.shape[0]),5)

# 원본(입력) 이미지를 맨 위에 그린다

for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28,1),cmap='gray')
    if i==0:
        ax.set_ylabel("INPUT",size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28,28,1),cmap='gray')
    if i==0:
        ax.set_ylabel("Noised",size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
for i, ax in enumerate([ax11,ax12,ax13,ax14,ax15]):
    ax.imshow(output[random_images[i]].reshape(28,28,1),cmap='gray')
    if i==0:
        ax.set_ylabel("Recursive",size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()
