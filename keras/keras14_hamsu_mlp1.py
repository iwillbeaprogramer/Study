# 다:1 mlp 함수형
# keras10_mlp2를 함수형으로 바꿔라
#다 : 1 mlp
import numpy as np
x = np.array([range(100),
             range(301,401),
             range(1,101)]).T
y = np.array(range(711,811))
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

# model = Sequential()
# model.add(Dense(20,input_dim=3))
# model.add(Dense(10))
# model.add(Dense(5))
# model.add(Dense(1))
inputs = Input(shape=(3,))
dense0 = Dense(20)(inputs)
dense1 = Dense(10)(dense0)
dense2 = Dense(5)(dense1)
outputs = Dense(1)(dense2)
model = Model(inputs=inputs,outputs=outputs)
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=300,batch_size=1,validation_split=0.2)


loss,mae = model.evaluate(x_test,y_test,batch_size=1)
print('loss : ',loss,'\tmae : ',mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
rmse = mean_squared_error(y_test,y_predict)**0.5
r2 = r2_score(y_test,y_predict)

print('rmse : ',rmse,'\tr2 : ',r2)


'''
loss :  0.00011988040205324069  mae :  0.00948181189596653
rmse :  0.010928565872370842    r2 :  0.9999998832470947
'''




