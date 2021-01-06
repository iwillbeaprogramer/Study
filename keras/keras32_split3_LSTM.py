# 과제및 실습
# early ,train_test_val, minmax,...  등등
# 데이터는 1~100/5개식 짜르기
#     x          y
# 1,2,3,4,5      6
# ...
# 95,...,99     100

# predict를 만들것
# 96,97,98,99,100 ->101
# ...
# 100,101,102,103,104 -> 105
# 예상 predict는 (101,102,103,104,105) with LSTM


import numpy as np
a = np.array(range(1,11))
size = 6

def split_x(seq,size):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size)]
        aaa.append(subset)
    return np.array(aaa)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

early_stopping = EarlyStopping(monitor='val_loss',patience=20,mode='auto')

ary = np.array(range(1,101))
datasets = split_x(ary,size)
x = datasets[:,:-1]
y = datasets[:,-1]

x_pred = split_x(np.array(range(96,106)),size)
x_pred = x_pred[:,:-1]


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_pred = scaler.transform(x_pred)

x_train=x_train.reshape(-1,5,1)
x_test=x_test.reshape(-1,5,1)
x_val=x_val.reshape(-1,5,1)
x_pred = x_pred.reshape(-1,5,1)
print(x_train)

'''
model = Sequential()
model.add(LSTM(128,activation='relu',input_shape=(5,1)))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1,))
model.compile(loss = 'mse',optimizer = 'adam')
model.fit(x_train,y_train,validation_data = (x_val,y_val),epochs=1000,callbacks=[early_stopping],batch_size=1)

loss = model.evaluate(x_test,y_test,batch_size = 1)
y_hat = model.predict(x_test)
y_pred = model.predict(x_pred)

print(y_pred)

rmse = mean_squared_error(y_hat,y_test)
r2 = r2_score(y_hat,y_test)

print("loss : ",loss)
print('r2 : ',r2)
print('rmse : ',rmse**0.5)
'''
'''
[[101.642494]
 [102.80078 ]
 [103.96667 ]
 [105.140114]
 [106.32112 ]]
loss :  0.11013299226760864
r2 :  0.9998669969236317
rmse :  0.33186389858021864
'''




