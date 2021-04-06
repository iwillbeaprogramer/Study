# 실습 vgg16으로 만들어봐

# 실습
# 남자 여자 구별
# ImageDataGenerator / fit 사용해서 완성    

import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

# train_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, width_shift_range=0.1, height_shift_range=0.1, rotation_range=5, zoom_range=1.2, shear_range=0.7, fill_mode='nearest',validation_split=0.2)
# test_datagen = ImageDataGenerator()
# xy_train = train_datagen.flow_from_directory("../data/image/male_female/male_female", target_size=(224,224), batch_size=50, class_mode='binary',subset='training')
# xy_val = train_datagen.flow_from_directory("../data/image/male_female/male_female", target_size=(224,224), batch_size=50, class_mode='binary',subset='validation')

# np.save('../data/image/male_female/k81_train_x.npy', arr=xy_train[0][0])
# np.save('../data/image/male_female/k81_train_y.npy', arr=xy_train[0][1])
# np.save('../data/image/male_female/k81_val_x.npy', arr=xy_val[0][0])
# np.save('../data/image/male_female/k81_val_y.npy', arr=xy_val[0][1])
x_train = np.load('../data/image/male_female/k81_train_x.npy')
y_train = np.load('../data/image/male_female/k81_train_y.npy')
x_val = np.load('../data/image/male_female/k81_val_x.npy')
y_val = np.load('../data/image/male_female/k81_val_y.npy')
print(x_train.shape,y_train.shape)

def callbacks():
    modelpath = "../data/image/male_female/models/best____.h5"
    es = EarlyStopping(monitor = 'val_loss',patience=30)
    cp = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath,save_best_only=True)
    reLR = ReduceLROnPlateau(monitor = 'val_loss',patience=8)
    return es,reLR,cp
es , reLR,cp = callbacks()




# 2. 모델
vgg16 = VGG16(weights = 'imagenet',include_top = False,input_shape=(224,224,3))
vgg16.trainable = False
model = Sequential()
model.add(vgg16)
model.add(Dense(1024,activation='relu',name = '123'))
model.add(Dense(1,activation='sigmoid',name = '1234'))
model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics=['accuracy'])
model.fit_generator(x_train,y_train,validation_data=(x_val,y_val),epochs = 500, callbacks = [es,reLR,cp])

