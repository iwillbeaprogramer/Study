# imagedatagenerator의 fit_generator 사용해서 완성
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,BatchNormalization,Flatten,Activation,Input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

train_datagen = ImageDataGenerator(rescale = 1./255, horizontal_flip=True, vertical_flip=True, width_shift_range=0.1, height_shift_range=0.1, rotation_range=5, zoom_range=1.2, shear_range=0.7, fill_mode='nearest',validation_split=0.2)
test_datagen = ImageDataGenerator(rescale = 1./255)

xy_train = train_datagen.flow_from_directory("../data/image/male_female/male_female", target_size=(100,100), batch_size=50, class_mode='binary',subset='training')
xy_val = train_datagen.flow_from_directory("../data/image/male_female/male_female", target_size=(100,100), batch_size=50, class_mode='binary',subset='validation')

def callbacks():
    modelpath = "../data/image/male_female/models/best.h5"
    es = EarlyStopping(monitor = 'val_loss',patience=30)
    cp = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath,save_best_only=True)
    reLR = ReduceLROnPlateau(monitor = 'val_loss',patience=8)
    return es,reLR,cp

def modeling():
    inputs = Input(shape=(100,100,3))
    x = inputs
    _x = Conv2D(64,3,padding='same')(x)
    _x = BatchNormalization()(_x)
    _x = Activation('relu')(_x)
    _x = Conv2D(64,3,padding='same')(_x)
    _x = BatchNormalization()(_x)
    _x = Activation('relu')(_x)
    x = _x
    _x = Conv2D(64,3,padding='same')(x)
    _x = BatchNormalization()(_x)
    _x = Activation('relu')(_x)
    x = x+_x
    x = MaxPooling2D(2)(x)
    _x = Conv2D(64,3,padding='same')(x)
    _x = BatchNormalization()(_x)
    _x = Activation('relu')(_x)
    x = x+_x
    x = MaxPooling2D(2)(x)
    _x = Conv2D(64,3,padding='same')(x)
    _x = BatchNormalization()(_x)
    _x = Activation('relu')(_x)
    x = x+_x
    x = MaxPooling2D(2)(x)
    x = Flatten()(x)
    x = Dense(2048)(x)
    x = Dense(1,activation='sigmoid')(x)
    outputs=x
    model = Model(inputs=inputs,outputs=outputs)
    return model

es , reLR,cp = callbacks()
'''
model = Sequential()
model.add(Conv2D(32,4,input_shape = (100,100,3),padding='valid'))
model.add(Conv2D(32,3,input_shape = (100,100,3),padding='valid'))
model.add(Conv2D(64,4,input_shape = (100,100,3),padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(32,3,padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
'''
model=modeling()
model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics=['accuracy'])
model.fit_generator(xy_train,validation_data=xy_val,epochs = 500, callbacks = [es,reLR,cp])


