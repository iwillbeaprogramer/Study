#다 : 1 mlp
import numpy as np
x = np.array([[1,2,3,4,5,6,7,8,9,10,],
             [11,12,13,14,15,16,17,18,19,20,]])
y = np.array([1,2,3,4,5,6,7,8,9,10,])

print(x.shape)
x=x.T
print(x.shape)
print(x)
print(x.shape)      # (2, 10) (10,)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10,input_dim=2))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x,y,epochs=100,batch_size=1,validation_split=0.2)

loss,mae = model.evaluate(x,y,batch_size=1)
print('loss : ',loss,'\tmae : ',mae)

y_predict = model.predict(x)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
rmse = mean_squared_error(y,y_predict)**0.5
r2 = r2_score(y,y_predict)

print('rmse : ',rmse,'\tr2 : ',r2)






