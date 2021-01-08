from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten

model = Sequential()
model.add(Conv2D(filters=10,kernel_size=(2,2),input_shape = (10,10,1)))  # ->> 넘어가면 (9,9,10)
model.add(Flatten())
model.add(Dense(1))
model.summary()