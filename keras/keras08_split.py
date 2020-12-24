import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

x = np.array(list(range(1,101)))
y = np.array(list(range(101,201)))
x_train = x[:60]
x_validation = x[60:80]
x_test = x[80:]
y_train = y[:60]
y_validation = y[60:80]
y_test = y[80:]


# y_train = np.array(list(range(1,11)))

# x_test = np.array(list(range(11,16)))
# y_test = np.array(list(range(11,16)))

# x_pred = np.array(list(range(16,19)))

model = Sequential()
model.add(Dense(10,input_dim = 1,activation='linear'))
model.add(Dense(5,activation='linear'))
model.add(Dense(1))

model.compile(loss = 'mae',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs = 1000,batch_size=1, validation_data=(x_validation,y_validation))

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
