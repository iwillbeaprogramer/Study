from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1 데이터
x = np.array(range(1,101))
y = np.array(range(1,101))

# x_train = x[:60]
# x_validation = x[60:80]
# x_test = x[80:]

# y_train = y[:60]
# y_validation = y[60:80]
# y_test = y[80:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7,test_size=0.2,shuffle=False)
#x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,train_size=0.8)

print('train size : ',x_train.shape,y_train.shape)
print('test size : ',x_test.shape,y_test.shape)
print(x_test)
#print('validation size : ',x_val.shape,y_val.shape)
'''
하나의 파라미터만 있을경우 나머지를 모두 배정하는식
둘다 있는데 1보다 크면 오류 1이면 위와같음
1보다 
train size :  (70,) (70,)
test size :  (20,) (20,)
'''





'''
model = Sequential()
model.add(Dense(100,input_dim=1))
model.add(Dense(50,activation='linear'))
model.add(Dense(10,activation='linear'))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=300,batch_size=1,validation_data=(x_val,y_val))

loss, mae = model.evaluate(x_test,y_test,batch_size=1)
print('loss : ',loss, 'mse : ',mae)
y_predict = model.predict(x_test)
#print(y_predict)
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)
print('loss : ',loss, '\nmse : ',mae,'\nR2 : ',r2)
#val x loss :  1.1263182608800548e-09 mse :  3.24249267578125e-05
#val o loss :  2.2395182895706967e-05 mse :  0.004373359493911266

#from sklearn.metrics import mean_squared_error
'''