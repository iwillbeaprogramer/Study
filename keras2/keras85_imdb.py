from tensorflow.keras.datasets import reuters, imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#1. DATA
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

print(x_train[0])
print(y_train[0:10])   # [1 0 0 1 0 0 1 0 1 0] >> 이진분류
print(x_train.shape, x_test.shape)  # (25000,) (25000,)
print(y_train.shape, y_test.shape)  # (25000,) (25000,)
print("==============================")

# x
print("최대 길이 : ", max(len(l) for l in x_train))         # 2494
print("평균 길이 : ",sum(map(len, x_train)) / len(x_train)) # 238.71364

word_to_index = imdb.get_word_index()
# print(word_to_index)
# print(type(word_to_index))

# y 카테고리 개수 출력
category = np.max(y_train) + 1 
print("y 카테고리 개수 :", category)    # 2

# preprocessing
# x
from tensorflow.keras.preprocessing.sequence import pad_sequences

max = 200
x_train = pad_sequences(x_train, maxlen=max, padding='pre')
x_test = pad_sequences(x_test, maxlen=max, padding='pre')
print(x_train.shape, x_test.shape)  # (25000, 200) (25000, 200)


#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Conv1D, Dropout

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=124, input_length=200))
model.add(Dropout(0.2))
model.add(Conv1D(128, 3, padding='valid', activation='relu',strides=1))
model.add(Dropout(0.2))
model.add(Conv1D(64, 3, padding='valid', activation='relu',strides=1))
model.add(LSTM(64))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

#3. Compile, Train
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.4, verbose=1)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=80, batch_size=16, validation_split=0.2)

#4. Evaluate, Predict
result = model.evaluate(x_test, y_test, batch_size=16)
print("loss : ", result[0])
print("acc : ", result[1])

# loss :  1.493330955505371
# acc :  0.8481600284576416