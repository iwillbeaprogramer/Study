# 실습 19_1번,2,3,4,5,6 earlystopping까지 총 6개 파일을 환성하시오

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import MinMaxScaler

datasets = load_diabetes()
x=datasets.data
y=datasets.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)


model=Sequential()
model.add(Dense(128,input_shape=(10,),activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=200,batch_size=1,verbose=1,validation_split=0.2)

loss = model.evaluate(x_test,y_test,batch_size=1)
y_predict = model.predict(x_test)
rmse = mean_squared_error(y_predict,y_test)
r2 = r2_score(y_predict,y_test)
print('loss : ',loss)
print('rmse : ',rmse)
print('R2 : ',r2)

'''
loss :  4253.93896484375
rmse :  4253.938913654165
R2 :  0.14423431063353742
'''



