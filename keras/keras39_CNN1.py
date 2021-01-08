from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten

model = Sequential()
model.add(Conv2D(filters=10,kernel_size=(2,2),padding='same',input_shape = (10,10,3),strides=2))  # ->> 넘어가면 (9,9,10)
model.add(Conv2D(9,(2,2)))
# model.add(Conv2D(9,(2,3)))
# model.add(Conv2D(8,2))
model.add(Flatten())
model.add(Dense(1))
model.summary()