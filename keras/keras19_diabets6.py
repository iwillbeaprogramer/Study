# 실습 19_1번,2,3,4,5,6 earlystopping까지 총 6개 파일을 환성하시오

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
np.random.seed(0)
#tf.set_random_seed(0)

datasets = load_diabetes()
x=datasets.data
y=datasets.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2)
x_scaler = MinMaxScaler()
x_scaler.fit(x_train)
x_train = x_scaler.transform(x_train)
x_test = x_scaler.transform(x_test)
x_val = x_scaler.transform(x_val)

early_stopping =EarlyStopping(monitor='val_loss',patience=20, mode = 'auto')

model=Sequential()
model.add(Dense(1000,input_shape=(10,)))
model.add(Dense(900,activation='relu'))
model.add(Dense(800,activation='relu'))
model.add(Dense(700,activation='relu'))
model.add(Dense(600,activation='relu'))
model.add(Dense(500,activation='relu'))
model.add(Dense(400,activation='relu'))
model.add(Dense(300,activation='relu'))
model.add(Dense(200,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(5,activation='relu'))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=2000,batch_size=4,verbose=1,validation_data=(x_val,y_val),callbacks=[early_stopping])

loss = model.evaluate(x_test,y_test,batch_size=4)
y_predict = model.predict(x_test)
rmse = mean_squared_error(y_predict,y_test)
r2 = r2_score(y_predict,y_test)
print('loss : ',loss)
print('rmse : ',rmse)
print('R2 : ',r2)

'''
loss :  17834.29296875
rmse :  17834.295936238046
R2 :  -1225563484693248.8
'''




