import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)                                                   # Train 에서는 Data가 많으면 좋기 때문에 증폭 사용
test_datagen = ImageDataGenerator(rescale=1./255)   # Test 에서는 Data를 증폭시킬 필요 X

# flow 또는 flow_from_directory
# 이미지 -> 데이터 화

# train_generator
xy_train = train_datagen.flow_from_directory(
    '../data/image/brain/train',
    target_size=(150,150),   # size 변경
    batch_size=5,            # batch_size 만큼 xy를 추출한다
    class_mode='binary'      # ad - Y 는 0 / normal - Y 는 1
)

# test_generator
xy_test = test_datagen.flow_from_directory(
    '../data/image/brain/test',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary'     
)

print(xy_train)     # <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001E79C798550>
print(xy_test)      # <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001E79CD10B20>

print(xy_train[0])
print(xy_train[0][0])   
print(xy_train[0][0].shape)   # X -> (5, 150, 150, 3)
print(xy_train[0][1])  
print(xy_train[0][1].shape)   # Y -> (5,)

# acc 올리기
model = Sequential()
model.add(Conv2D(36, (3,3), padding='same', activation='relu', input_shape=(150, 150, 3)))
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit_generator(
    xy_train, steps_per_epoch=32, epochs=200,
    validation_data=xy_test, validation_steps=4
)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# 시각화 할 것!!

plt.plot(acc)
plt.plot(val_acc)
plt.legend(['acc', 'val_acc'])
plt.show()

print('acc :', acc[-1])
print('val_acc :', val_acc[-1])