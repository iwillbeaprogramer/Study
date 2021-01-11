# 실습 19_1번,2,3,4,5,6 earlystopping까지 총 6개 파일을 환성하시오

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import tensorflow as tf
np.random.seed(0)
#tf.set_random_seed(0)

datasets = load_diabetes()
x=datasets.data
y=datasets.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=45)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2,random_state=45)
print(x_test[0:5,:])

x_scaler = MinMaxScaler()
x_scaler.fit(x_train)
x_train = x_scaler.transform(x_train)
x_test = x_scaler.transform(x_test)
x_val = x_scaler.transform(x_val)
print(x_test[0:5,:])
modelpath = './modelCheckpoint/k46_diabets_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True)
early_stopping =EarlyStopping(monitor='val_loss',patience=70)

model=Sequential()
model.add(Dense(128,activation='relu',input_shape=(10,)))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=20000,batch_size=4,verbose=1,validation_data=(x_val,y_val),callbacks=[early_stopping,cp])

loss = model.evaluate(x_test,y_test,batch_size=4)
y_predict = model.predict(x_test)
rmse = mean_squared_error(y_predict,y_test)**0.5
r2 = r2_score(y_predict,y_test)
print('loss : ',loss)
print('rmse : ',rmse)
print('R2 : ',r2)

'''
loss :  17834.29296875
rmse :  17834.295936238046
R2 :  -1225563484693248.8
'''




