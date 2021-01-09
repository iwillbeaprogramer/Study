# 사이킷런 데이터셋
# LSTM모델링
# Dense와 성능비교

from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
from tensorflow.keras.callbacks import EarlyStopping


early_stopping = EarlyStopping(monitor = 'loss',mode='auto',patience=20)
datasets = load_boston()
x = datasets.data
y = datasets.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

x_train = x_train.reshape(-1,13,1,1)
x_test = x_test.reshape(-1,13,1,1)
x_val = x_val.reshape(-1,13,1,1)

model = Sequential()
model.add(Conv2D(filters = 128,kernel_size=(2,1),input_shape = (13,1,1)))
model.add(MaxPooling2D(pool_size=1))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))
model.compile(loss = 'mse',optimizer='adam')
model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=2000,verbose=1,callbacks=[early_stopping],batch_size=4)

loss = model.evaluate(x_test,y_test,batch_size=4,verbose=1)
y_pred = model.predict(x_test)
r2 = r2_score(y_pred,y_test)
rmse = mean_squared_error(y_pred,y_test)**0.5

print('rmse : ',rmse)
print('r2 : ',r2)



'''
DNN
loss :  6.71339750289917
rmse :  6.713397844367864
r2_Score :  0.9213530563370745


LSTM
rmse :  3.8205305080023364
r2 :  0.8176361801219632

CNN
rmse :  3.9031809150322885
r2 :  0.6623068498248086
'''