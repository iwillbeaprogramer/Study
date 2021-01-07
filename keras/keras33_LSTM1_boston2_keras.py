# 케라스 데이터셋
# LSTM모델링
# Dense와 성능비교

# 사이킷런 데이터셋
# LSTM모델링
# Dense와 성능비교

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import boston_housing


early_stopping = EarlyStopping(monitor = 'val_loss',mode='auto',patience=10)
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)

model = Sequential()
model.add(LSTM(128,input_shape = (x_train.shape[1],x_train.shape[2]),activation='relu'))
model.add(Dense(128,input_shape=(13,),activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))
model.compile(loss = 'mse',optimizer='adam')
model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=2000,verbose=1,callbacks=[early_stopping],batch_size=2)

loss = model.evaluate(x_test,y_test,batch_size=2,verbose=1)
y_pred = model.predict(x_test)
r2 = r2_score(y_pred,y_test)
rmse = mean_squared_error(y_pred,y_test)**0.5

print('rmse : ',rmse)
print('r2 : ',r2)


'''
DNN
rmse :  5.373893230328938
r2 :  0.5467260151385238

rmse :  5.12852377573834
r2 :  0.6618843434433714
'''