# 다:다
# 실습

import numpy as np
x = np.array([range(100),
             range(301,401),
             range(1,101),
             range(1200,1500,3),
             range(2000,3000,10)]).T
y = np.array([range(711,811),range(201,301)]).T
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
print(x_train.shape)
print(y_train.shape)

x_pred2=np.array([100,402,101,100,401])
x_pred2 = x_pred2.reshape(1,5)

model = Sequential()
model.add(Dense(5000,input_dim=5))
model.add(Dense(10))
model.add(Dense(1000))
model.add(Dense(10))
model.add(Dense(1000))
model.add(Dense(10))
model.add(Dense(1000))
model.add(Dense(10))
model.add(Dense(1000))
model.add(Dense(10))
model.add(Dense(1000))
model.add(Dense(10))
model.add(Dense(1000))
model.add(Dense(2))
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=30,batch_size=8,validation_split=0.2)


loss,mae = model.evaluate(x_test,y_test,batch_size=8)
print('loss : ',loss,'\tmae : ',mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
rmse = mean_squared_error(y_test,y_predict)**0.5
r2 = r2_score(y_test,y_predict)
print('rmse : ',rmse,'\tr2 : ',r2)

for i in range(5,10):
    print('test : ',y_test[i],'predict : ',y_predict[i])





