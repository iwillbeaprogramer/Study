# 2개의 파일을 만들어서
# earlystopping을 적용하지 않은 최고의 모델
# earlystopping을 적용한 최고의 모델

from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2)
#scaler = MinMaxScaler()
#scaler.fit(x_train)
#x_test = scaler.transform(x_test)
#x_val = scaler.transform(x_val)

model = Sequential()
model.add(Dense(128,activation='relu',input_shape=(13,)))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1))
model.compile(loss='mse',optimizer = 'adam')
model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=200,batch_size=2,verbose=1,)

loss = model.evaluate(x_test,y_test,batch_size=2)
y_prediction = model.predict(x_test)
r2 = r2_score(y_prediction,y_test)
rmse = mean_squared_error(y_prediction,y_test)**0.5
print('rmse : ',rmse)
print('r2 : ',r2)

for i in range(10):
    label = y_test[i]
    prediction = y_prediction[i]
    print("실제가격 : {}, 예상가격 : {}".format(label,prediction))


'''
rmse :  5.654013456980459
r2 :  0.5397004208245597
실제가격 : 7.2, 예상가격 : [8.451794]
실제가격 : 18.8, 예상가격 : [17.6656]
실제가격 : 19.0, 예상가격 : [20.266815]
실제가격 : 27.0, 예상가격 : [35.17088]
실제가격 : 22.2, 예상가격 : [21.362865]
실제가격 : 24.5, 예상가격 : [19.859428]
실제가격 : 31.2, 예상가격 : [27.509365]
실제가격 : 22.9, 예상가격 : [21.661318]
실제가격 : 20.5, 예상가격 : [18.821333]
실제가격 : 23.2, 예상가격 : [18.635452]
'''