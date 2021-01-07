# 실습 19_1번,2,3,4,5,6 earlystopping까지 총 6개 파일을 환성하시오

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score,mean_squared_error

datasets = load_diabetes()
x, y = datasets.data,datasets.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


model = Sequential()
model.add(Dense(128,input_dim=10,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,validation_split=0.2,epochs=200,batch_size=8,verbose=1)

loss = model.evaluate(x_test,y_test,batch_size=8,verbose=0)
print('loss : ',loss)
y_predict = model.predict(x_test)
rmse = mean_squared_error(y_predict,y_test)**0.5
r2 = r2_score(y_predict,y_test)
print('rmse : ',rmse)
print('r2_Score : ',r2)

'''
loss :  3204.5146484375
rmse :  3204.5142793344044
R2 :  0.21290600730061382
'''