# early stopping

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston


x = np.load('./data/boston_x.npy')
y = np.load('./data/boston_y.npy')
#dataset = load_boston()
#x, y = dataset.data,dataset.target

#print(np.max(x),np.min(x))
#print(dataset.feature_names)
#print(dataset.DESCR)

from sklearn.preprocessing import MinMaxScaler
print(np.max(x),np.min(x))



from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
modelpath = './modelCheckpoint/k46_boston_{epoch:02d}-{val_loss:.4f}.hdf5'
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 20,mode = 'auto')
cp = ModelCheckpoint(filepath=modelpath,monitor='val_loss',mode='auto')

model = Sequential()
model.add(Dense(128,input_shape=(13,),activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
hist = model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=2000,batch_size=4,verbose=1,callbacks=[early_stopping,cp])

loss = model.evaluate(x_test,y_test,batch_size=4,verbose=0)
print('loss : ',loss)
y_predict = model.predict(x_test)
rmse = mean_squared_error(y_predict,y_test)
r2 = r2_score(y_predict,y_test)
print('rmse : ',rmse)
print('r2_Score : ',r2)