# 실습
# 남자 여자 구별
# ImageDataGenerator / fit 사용해서 완성    

import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest',
    validation_split=0.2
)
test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2) 

# male - 841 (1) , female - 895 (0)
xy_train = train_datagen.flow_from_directory(
    '../data/image/human',
    target_size=(150,150),
    batch_size=1389,
    class_mode='binary',
    shuffle=True,
    subset='training'
)

xy_val = train_datagen.flow_from_directory(
    '../data/image/human',
    target_size=(150,150),
    batch_size=347,
    class_mode='binary',
    shuffle=False,
    subset='validation'
)

print(xy_train[0])
print(xy_train[0][0])   
print(xy_train[0][0].shape)   # X -> (1389, 150, 150, 3)
print(xy_train[0][1])  
print(xy_train[0][1].shape)   # Y -> (1389,)

print(xy_val[0])
print(xy_val[0][0])   
print(xy_val[0][0].shape)     # X -> (347, 150, 150, 3)
print(xy_val[0][1])  
print(xy_val[0][1].shape)     # Y -> (347,)

np.save('../data/image/brain/npy/k67_train_x.npy', arr=xy_train[0][0])
np.save('../data/image/brain/npy/k67_train_y.npy', arr=xy_train[0][1])
np.save('../data/image/brain/npy/k67_val_x.npy', arr=xy_val[0][0])
np.save('../data/image/brain/npy/k67_val_y.npy', arr=xy_val[0][1])

x_train = np.load('../data/image/brain/npy/k67_train_x.npy')
y_train = np.load('../data/image/brain/npy/k67_train_y.npy')
x_val = np.load('../data/image/brain/npy/k67_val_x.npy')
y_val = np.load('../data/image/brain/npy/k67_val_y.npy')

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)

# 2. 모델
model = Sequential()
model.add(Conv2D(256, (3,3), padding='same', activation='relu', input_shape=(150,150,3)))
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=3))
model.add(Dropout(0.2))
model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=3))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_val, y_val), verbose=2)