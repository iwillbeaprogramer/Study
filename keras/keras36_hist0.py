import numpy as np
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,LSTM

#1 data
a = np.array(range(1,101))
size=5

def split_x(seq,size):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size)]
        aaa.append(subset)
    return np.array(aaa)

datasets = split_x(a,size)
print(datasets.shape)
x=datasets[:,0:4]
x=x.reshape(x.shape[0],x.shape[1],1)
y=datasets[:,-1]

print(x.shape,y.shape)

# 2
model = load_model('./model/save_keras35.h5')
model.add(Dense(5,name='123123'))
model.add(Dense(1,name='1231234'))

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=10)

model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
hist = model.fit(x,y,epochs=1000,batch_size=32,verbose=1,validation_split=0.2,callbacks=[es])
print(hist)
print(hist.history.keys())

print(hist.history['loss'])

import matplotlib.pyplot as plt
plt.plot(hist.history['loss'][-20:-1])
plt.plot(hist.history['val_loss'][-20:-1])
#plt.plot(hist.history['accuracy'])
#plt.plot(hist.history['val_accuracy'])
plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epochs')
plt.legend(['train_loss','val_loss','train_acc','val_acc'])
plt.show()