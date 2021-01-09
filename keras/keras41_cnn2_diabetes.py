# 사이킷런 데이터셋
# LSTM모델링
# Dense와 성능비교
# 회귀모델

# 실습 19_1번,2,3,4,5,6 earlystopping까지 총 6개 파일을 환성하시오

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Conv2D,MaxPooling2D,Flatten
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

x_train = x_train.reshape(x_train.shape[0],2,5,1)
x_test = x_test.reshape(x_test.shape[0],2,5,1)
x_val = x_val.reshape(x_val.shape[0],2,5,1)

early_stopping =EarlyStopping(monitor='loss',patience=70)
model=Sequential()
model.add(Conv2D(128,kernel_size=2,padding='valid',input_shape = (2,5,1)))
model.add(MaxPooling2D(pool_size=1))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=20000,batch_size=4,verbose=1,validation_data=(x_val,y_val),callbacks=[early_stopping])

loss = model.evaluate(x_test,y_test,batch_size=4)
y_predict = model.predict(x_test)
rmse = mean_squared_error(y_predict,y_test)**0.5
r2 = r2_score(y_predict,y_test)
print('loss : ',loss)
print('rmse : ',rmse)
print('R2 : ',r2)


'''
DNN
loss :  17834.29296875
rmse :  17834.295936238046
R2 :  -1225563484693248.8

LSTM
rmse :  4319.449432814543
R2 :  0.2393480819718682

CNN
rmse :  58.605869760966236
R2 :  -0.5028310388938366
'''




