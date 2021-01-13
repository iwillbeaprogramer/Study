# 사이킷런 데이터셋
# LSTM모델링
# Dense와 성능비교
# 이진분류

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Conv1D,Flatten,MaxPooling1D,Dropout
from tensorflow.keras.callbacks import EarlyStopping

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2,shuffle=True)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)

early_stopping = EarlyStopping(monitor='val_loss',patience=25,mode='auto')


model = Sequential()
model.add(Conv1D(filters = 128,kernel_size=2,input_shape = (x_train.shape[1],x_train.shape[2]),activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=10000,verbose=1,callbacks=[early_stopping],batch_size=2)

loss = model.evaluate(x_test,y_test,batch_size=2)
y_prediction = model.predict_classes(x_test)

count=0
for i in range(len(y_test)):
    print('실제 : {} , 예측 : {}'.format(y_test[i],y_prediction[i][0]))
    if y_test[i] != y_prediction[i][0]:
        count+=1
        print('여기다')
print('loss : ',loss[1])

'''
DNN
accuracy :  0.9824561476707458

LSTM
loss :  0.9561403393745422

Conv1d
acc : 0.5789
'''
