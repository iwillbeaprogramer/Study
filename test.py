import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Conv2D,BatchNormalization,Dropout,Activation,LeakyReLU,UpSampling2D,Input,Dense,Reshape,Flatten,Conv2DTranspose,ReLU,concatenate,ZeroPadding2D
import numpy as np
def Generator(basic_filters=64,kernel_size=4,drop_out=0.5):
    
    initializer = tf.random_normal_initializer(0.,0.02)
    inputs = Input(shape=(720,1280,3))
    layer1 = Conv2D(filters = basic_filters,kernel_size=4,strides=2,padding='same',use_bias=False,kernel_initializer=initializer)(inputs)
    layer1 = LeakyReLU()(layer1)
    layer1_ = layer1
    
    layer2 = Conv2D(filters=basic_filters*2,kernel_size=kernel_size,strides=2,padding='same',use_bias=False,kernel_initializer=initializer)(layer1)
    layer2_ = BatchNormalization()(layer2)
    layer2 = LeakyReLU()(layer2_)
    
    layer3 = Conv2D(filters=basic_filters*4,kernel_size=kernel_size,strides=2,padding='same',use_bias=False,kernel_initializer=initializer)(layer2)
    layer3_ = BatchNormalization()(layer3)
    layer3 = LeakyReLU()(layer3_)
    
    layer4 = Conv2D(filters=basic_filters*8,kernel_size=kernel_size,strides=(2,2),padding='same',use_bias=False,kernel_initializer=initializer)(layer3)
    layer4_ = BatchNormalization()(layer4)
    layer4 = LeakyReLU()(layer4_)
    
    layer5 = Conv2D(filters=basic_filters*8,kernel_size=kernel_size,strides=(3,2),padding='same',use_bias=False,kernel_initializer=initializer)(layer4)
    layer5_ = BatchNormalization()(layer5)
    layer5 = LeakyReLU()(layer5_)
    
    layer6 = Conv2D(filters=basic_filters*8,kernel_size=kernel_size,strides=(3,2),padding='same',use_bias=False,kernel_initializer=initializer)(layer5)
    layer6_ = BatchNormalization()(layer6)
    layer6 = LeakyReLU()(layer6_)
    
    layer7 = Conv2D(filters=basic_filters*8,kernel_size=kernel_size,strides=(1,2),padding='same',use_bias=False,kernel_initializer=initializer)(layer6)
    layer7_ = BatchNormalization()(layer7)
    layer7 = LeakyReLU()(layer7_)
    
    layer8 = Conv2D(filters=basic_filters*16,kernel_size=kernel_size,strides=(1,2),padding='same',use_bias=False,kernel_initializer=initializer)(layer7)
    layer8_ = BatchNormalization()(layer8)
    layer8 = LeakyReLU()(layer8_)
    
    # 가운데
    layer9 = Conv2D(filters=basic_filters*16,kernel_size=kernel_size,strides=(5,5),padding='same',use_bias=False,kernel_initializer=initializer)(layer8)
    layer9_ = BatchNormalization()(layer9)
    layer9 = LeakyReLU()(layer9_)
    # 가운데
    
    layer10 = Conv2DTranspose(filters=basic_filters*16,kernel_size=kernel_size,strides=(5,5),padding='same',kernel_initializer=initializer,use_bias=False)(layer9)
    layer10 = BatchNormalization()(layer10)
    layer10 = layer10+layer8_
    layer10 = Dropout(drop_out)(layer10)
    layer10 = ReLU()(layer10)
    
    layer11 = Conv2DTranspose(filters=basic_filters*8,kernel_size=kernel_size,strides=(1,2),padding='same',kernel_initializer=initializer,use_bias=False)(layer10)
    layer11 = BatchNormalization()(layer11)
    layer11 = layer11+layer7_
    layer11 = Dropout(drop_out)(layer11)
    layer11 = ReLU()(layer11)
    
    layer12 = Conv2DTranspose(filters=basic_filters*8,kernel_size=kernel_size,strides=(1,2),padding='same',kernel_initializer=initializer,use_bias=False)(layer11)
    layer12 = BatchNormalization()(layer12)
    layer12 = layer12+layer6_
    layer12 = Dropout(drop_out)(layer12)
    layer12 = ReLU()(layer12)
    
    layer13 = Conv2DTranspose(filters=basic_filters*8,kernel_size=kernel_size,strides=(3,2),padding='same',kernel_initializer=initializer,use_bias=False)(layer12)
    layer13 = BatchNormalization()(layer13)
    layer13 = layer13+layer5_
    layer13 = Dropout(drop_out)(layer13)
    layer13 = ReLU()(layer13)
    
    layer14 = Conv2DTranspose(filters=basic_filters*8,kernel_size=kernel_size,strides=(3,2),padding='same',kernel_initializer=initializer,use_bias=False)(layer13)
    layer14 = BatchNormalization()(layer14)
    layer14 = layer14+layer4_
    layer14 = Dropout(drop_out)(layer14)
    layer14 = ReLU()(layer14)
    
    layer15 = Conv2DTranspose(filters=basic_filters*4,kernel_size=kernel_size,strides=(2,2),padding='same',kernel_initializer=initializer,use_bias=False)(layer14)
    layer15 = BatchNormalization()(layer15)
    layer15 = layer15+layer3_
    layer15 = Dropout(drop_out)(layer15)
    layer15 = ReLU()(layer15)
    
    layer16 = Conv2DTranspose(filters=basic_filters*2,kernel_size=kernel_size,strides=(2,2),padding='same',kernel_initializer=initializer,use_bias=False)(layer15)
    layer16 = BatchNormalization()(layer16)
    layer16 = layer16+layer2_
    layer16 = Dropout(drop_out)(layer16)
    layer16 = ReLU()(layer16)
    
    layer17 = Conv2DTranspose(filters=basic_filters,kernel_size=kernel_size,strides=(2,2),padding='same',kernel_initializer=initializer,use_bias=False)(layer16)
    layer17 = BatchNormalization()(layer17)
    layer17 = layer17+layer1_
    layer17 = Dropout(drop_out)(layer17)
    layer17 = ReLU()(layer17)
    
    outputs = Conv2DTranspose(filters=basic_filters,kernel_size=kernel_size,strides=(2,2),padding='same',kernel_initializer=initializer,use_bias=False)(layer16)
    outputs = BatchNormalization()(outputs)
    outputs = Dropout(drop_out)(outputs)
    outputs = ReLU()(outputs)
    
    outputs = Conv2DTranspose(3,4,strides=2,padding='same',kernel_initializer=initializer,activation='tanh',name = 'last')(outputs)    
    
    model = Model(inputs=inputs,outputs=outputs)
    
    return model

model = Generator()
model.summary()

model.get_layer(name='last').trainable=False

model.summary()