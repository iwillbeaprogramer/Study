import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

x_train = np.load('../data/image/brain/npy/k66_train_x.npy')
y_train = np.load('../data/image/brain/npy/k66_train_y.npy')
x_test = np.load('../data/image/brain/npy/k66_test_x.npy')
y_test = np.load('../data/image/brain/npy/k66_test_y.npy')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# 실습
# 모델을 만들어라

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3,3), input_shape=(150,150,3)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=32, validation_split=0.2, verbose=2)
