from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2
)

# num_words=10000 : 10000번째까지 단어들만 사용한다.

# 문장을 자르고 수치화까지 한 데이터
print(x_train[0], type(x_train[0]))   # [1, 2, 2, 8, 43, 10 ... 132, 6, 109, 15, 17, 12] <class 'list'> 하나의 문장이 들어있는 한 리스트
print(y_train[0])   # 3
print(len(x_train[0]), len(x_train[11]))    # 87, 59 >> 문장의 길이가 다르다.
print("============================================================")
print(x_train.shape, x_test.shape)  # (8982,) (2246,)
print(y_train.shape, y_test.shape)  # (8982,) (2246,)


# x
print("뉴스기사 최대 길이 : ", max(len(l) for l in x_train))            # 2376
print("뉴스기사 평균 길이 : ", sum(map(len, x_train)) / len(x_train))   # 145.5398574927633
# map : 원래 for문을 사용해서 작업해야 할 것을 map 한 줄로 한꺼번에 처리할 수 있다.
# map(len, x_train) : x_train 리스트에 있는 모든 요소의 len을 반환한다.

# plt.hist([len(s) for s in x_train], bins=50)    # x 데이터의 길이
# plt.show()

# y 
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)  # (8982, 46) (2246, 46)


# y 분포
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("y 분포 : ", dict(zip(unique_elements, counts_elements)))
# {0: 55, 1: 432, 2: 74, 3: 3159, 4: 1949, 5: 17, 6: 48, 7: 16, 8: 139, 9: 101, 10: 124, 11: 390, 12: 49, 
# 13: 172, 14: 26, 15: 20, 16: 444, 17: 39, 18: 66, 19: 549, 20: 269, 21: 100, 22: 15, 23: 41, 24: 62, 25: 92, 
# 26: 24, 27: 15, 28: 48, 29: 19, 30: 45, 31: 39, 32: 32, 33: 11, 34: 50, 35: 10, 36: 49, 37: 19, 38: 19, 39: 24, 
# 40: 36, 41: 30, 42: 13, 43: 21, 44: 12, 45: 18} >> 총 46개의 카테고리
print("=================================")

# plt.hist(y_train, bins=46)    # bins : 가로를 bins로 나눈 값으로 히스토그램의 막대 너비가 정해진다.
# plt.show()

# x 단어들 분포
word_to_index = reuters.get_word_index()
print(word_to_index)
print(type(word_to_index))  # 해당 인덱스의 개수 == input_dim
# {'mdbl': 10996, ...
print("=================================")

# key와 value 위치를 교체 >> 원 문장이 뭔지 알아보자
index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value] = key

# key와 value 위치를 교체 후
print(index_to_word)
# {10996: 'mdbl' ...

print(len(index_to_word))   # 30979

print(index_to_word[1])     # 가장 빈도수가 많은 단어 >> the
print(index_to_word[30979]) # 가장 빈도수가 작은 단어 >> northerly


# x_train[0] 문장으로 복원시켜본다.
print(x_train[0])
# print(' '.join([index_to_word[index] for index in x_train[0]]))
# the wattie nondiscriminatory mln loss for plc said at only ended said commonwealth could 1 traders now april 0 a after said from 1985 and from foreign 000 april 0 prices its account year a but in this mln home an states earlier and rise and revs vs 000 its 16 vs 000 a but 3 psbr oils several and shareholders and dividend vs 000 its all 4 vs 000 1 mln agreed largely april 0 are 2 states will billion total and against 000 pct dlrs  


from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x_train = pad_sequences(x_train, padding='pre', maxlen=250)
pad_x_test = pad_sequences(x_test, padding='pre', maxlen=250)

print(pad_x_train)
print(pad_x_train.shape)    
print(pad_x_test.shape)    

print(len(np.unique(pad_x_train)))  # 9908

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

model = Sequential()

model.add(Embedding(input_dim=10000, output_dim=150, input_length=250))
model.add(LSTM(128))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(46, activation='softmax'))

model.summary()

# Compile, Train

es = EarlyStopping(monitor='val_loss', patience=30, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=15, mode='min')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit(pad_x_train, y_train, batch_size=8, epochs=50, validation_split=0.1, callbacks=[es, lr])

loss, acc = model.evaluate(pad_x_test, y_test)
print("loss : ", loss)
print("acc : ", acc)

# loss :  2.330373525619507
# acc :  0.7284060716629028