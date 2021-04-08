# keras67_1 남자 여자에 noise를 넣어서
# 기미 주근깨 여드름을 제거하시오

# 실습
# 남자 여자 구별
# ImageDataGenerator / fit 사용해서 완성    

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,MaxPooling2D,BatchNormalization,Dropout,LeakyReLU,Input,ReLU
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

cp = ModelCheckpoint(monitor='val_loss',filepath = './male_female_model.h5',save_best_only=True)

def custom_loss(y_true,y_pred):
    return tf.math.reduce_mean(tf.square(y_true-y_pred))-0.1*tf.math.reduce_std(y_pred)

# 1. 데이터
x_train = (np.load('../data/image/brain/npy/k67_train_x__.npy')/255.0 - 0.5)/0.5
x_train,x_test = train_test_split(x_train,test_size=0.2)

# print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

x_train_noised = x_train + np.random.normal(0,0.2,size=x_train.shape)
x_test_noised = x_test + np.random.normal(0,0.2,size=x_test.shape)
x_train_noised = np.clip(x_train_noised,a_min = -1,a_max=1)
x_test_noised = np.clip(x_test_noised,a_min=-1,a_max=1)

def autoencoder():
    inputs = Input(shape = (224,224,3))
    layer1 = Conv2D(filters=64,kernel_size=4,strides=2,use_bias=False,padding='same')(inputs)
    layer1 = LeakyReLU()(layer1)
    layer1_ = layer1

    layer2 = Conv2D(filters=128,kernel_size=4,strides=2,use_bias=False,padding='same')(layer1)
    layer2_ = BatchNormalization()(layer2)
    layer2 = LeakyReLU()(layer2_)
    
    layer3 = Conv2D(filters=256,kernel_size=4,strides=2,use_bias=False,padding='same')(layer2)
    layer3_ = BatchNormalization()(layer3)
    layer3 = LeakyReLU()(layer3_)

    layer4 = Conv2D(filters=512,kernel_size=4,strides=2,use_bias=False,padding='same')(layer3)
    layer4_ = BatchNormalization()(layer4)
    layer4 = LeakyReLU()(layer4_)



    layer5 = Conv2D(filters=512,kernel_size=4,strides=2,use_bias=False,padding='same')(layer4)
    layer5_ = BatchNormalization()(layer5)
    layer5 = LeakyReLU()(layer5_)



    layer6 = Conv2DTranspose(filters=512,kernel_size=4,strides=2,use_bias=False,padding='same')(layer5)
    layer6 = BatchNormalization()(layer6)
    layer6 = layer6+layer4_
    layer6 = Dropout(0.5)(layer6)
    layer6 = ReLU()(layer6)

    layer7 = Conv2DTranspose(filters=256,kernel_size=4,strides=2,use_bias=False,padding='same')(layer6)
    layer7 = BatchNormalization()(layer7)
    layer7 = layer7+layer3_
    layer7 = Dropout(0.5)(layer7)
    layer7 = ReLU()(layer7)

    layer8 = Conv2DTranspose(filters=128,kernel_size=4,strides=2,use_bias=False,padding='same')(layer7)
    layer8 = BatchNormalization()(layer8)
    layer8 = layer8+layer2_
    layer8 = Dropout(0.5)(layer8)
    layer8 = ReLU()(layer8)

    layer9 = Conv2DTranspose(filters=64,kernel_size=4,strides=2,use_bias=False,padding='same')(layer8)
    layer9 = BatchNormalization()(layer9)
    layer9 = layer9+layer1_
    layer9 = Dropout(0.5)(layer9)
    layer9 = ReLU()(layer9)

    outputs = Conv2DTranspose(3,kernel_size=3,strides=2,padding='same',activation='tanh')(layer9)
    model = Model(inputs=inputs,outputs=outputs)
    return model

model = autoencoder()
# model.summary()

model.compile(loss = custom_loss,optimizer = 'adam')
model.fit(x_train_noised,x_train,epochs=50,validation_split=0.1,batch_size=128,callbacks=[cp])

model = load_model('./male_female_model.h5',compile=False)
output = (model.predict(x_test_noised))

from matplotlib import pyplot as plt
import random
fig,((ax1,ax2,ax3,ax4,ax5),(ax6,ax7,ax8,ax9,ax10),(ax11,ax12,ax13,ax14,ax15)) = plt.subplots(3,5,figsize=(20,7))

# 이미지 5개를 무작위로 고른다

random_images = random.sample(range(output.shape[0]),5)

# 원본(입력) 이미지를 맨 위에 그린다

for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
    ax.imshow(x_test[random_images[i]].reshape(224,224,3))
    if i==0:
        ax.set_ylabel("INPUT",size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(224,224,3))
    if i==0:
        ax.set_ylabel("Noised",size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
for i, ax in enumerate([ax11,ax12,ax13,ax14,ax15]):
    ax.imshow(output[random_images[i]].reshape(224,224,3))
    if i==0:
        ax.set_ylabel("Recursive",size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()