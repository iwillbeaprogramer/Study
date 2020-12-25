#실습
#R2를 음수가 아닌 0.5 이하로 줄이기
#1.레이어는 인풋과 아웃품을 포함 5개 이상
#2.batch_size=1
#3.epochs=100이상
#4.데이터 조작 금지


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

x_train = np.array(list(range(1,11)))
y_train = np.array(list(range(1,11)))

x_test = np.array(list(range(11,16)))
y_test = np.array(list(range(11,16)))

x_pred = np.array(list(range(16,19)))

model = Sequential()
model.add(Dense(3000,input_dim = 1,activation='linear'))
model.add(Dense(30))
model.add(Dense(3000))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(2000))
model.add(Dense(20))
model.add(Dense(2000))
model.add(Dense(20))
model.add(Dense(2000))
model.add(Dense(1))

model.compile(loss = 'mae',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs = 100,batch_size=1, validation_split=0.2)

results = model.evaluate(x_test,y_test,batch_size=1)
print("mse, mae : ", results)
y_predict = model.predict(x_test)
print("y_predict : ",y_predict)

from sklearn.metrics import mean_absolute_error

def RMSE(y_test,y_predict):
    return np.sqrt(mean_absolute_error(y_test,y_predict))

print("RMSE : ", RMSE(y_test,y_predict))
print('mse : ',mean_absolute_error(y_test,y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)
print("R2 : ",r2)