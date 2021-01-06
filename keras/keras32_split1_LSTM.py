import numpy as np

def split_x(seq,size):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size)]
        aaa.append(subset)
    return np.array(aaa)

a= np.array(range(1,11))
size = 5

datasets = split_x(a,size)
x = datasets[:,0:-1]
y = datasets[:,-1]
x = x.reshape(x.shape[0],x.shape[1],1)

from tensorflow.keras.layers import LSTM,Dense
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(LSTM(16,input_shape=(x.shape[1],x.shape[2]),activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=500,batch_size=1)

loss = model.evaluate(x,y,batch_size=1)
print('loss : ',loss)


'''
loss :  0.0011558582773432136
'''