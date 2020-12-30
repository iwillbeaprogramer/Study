# 1:다
import numpy as np
x = np.array([range(100)]).T
y = np.array([range(711,811),range(1,101),range(201,301)]).T
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
print(x_train.shape)
print(y_train.shape)
x_pred2 = np.array([100,101,102,103,104]).T


inputs = Input(shape=(1,))
dense0 = Dense(100)(inputs)
dense1 = Dense(50)(dense0)
dense2 = Dense(25)(dense1)
dense3 = Dense(5)(dense2)
outputs = Dense(3)(dense3)
model = Model(inputs=inputs,outputs=outputs)
# model = Sequential()
# model.add(Dense(20,input_dim=3))
# model.add(Dense(10))
# model.add(Dense(5))
# model.add(Dense(3))
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=400,batch_size=1,validation_split=0.2)


loss,mae = model.evaluate(x_test,y_test,batch_size=1)
print('loss : ',loss,'\tmae : ',mae)
y_predict = model.predict(x_test)
y_predict2 = model.predict(x_pred2)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
rmse = mean_squared_error(y_test,y_predict)**0.5
r2 = r2_score(y_test,y_predict)

print('rmse : ',rmse,'\tr2 : ',r2)
print(y_predict2)




