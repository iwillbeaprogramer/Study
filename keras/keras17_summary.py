import numpy as np
import tensorflow as tf

x = np.array([1,2,3,4,5,6,7,8,9,10,]).reshape(-1,1)
y = np.array([2,4,6,8,10,12,14,16,18,20]).reshape(-1,1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(7,input_dim=2,activation='linear'))
model.add(Dense(5,activation='linear',name='aaa'))
model.add(Dense(8,activation='linear',name='aaa'))
model.add(Dense(10))
model.summary()

# 실습2 + 과제
# ensemble 1,2,3,4 에 대해 서머리 파람 계산
# layer을 만들때 이름을 설명할것?