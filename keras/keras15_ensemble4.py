# 다:다
import numpy as np
x1 = np.array([range(100),range(301,401),range(1,101)]).T
y1 = np.array([range(711,811),range(1,101),range(201,301)]).T
y2 = np.array([range(501,601),range(711,811),range(100)]).T

from sklearn.model_selection import train_test_split
x1_train,x1_test,y1_train,y1_test,y2_train,y2_test = train_test_split(x1,y1,y2,test_size=0.2,shuffle=False)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input,Dense,LSTM,Conv2D,Concatenate,concatenate

inputs1 = Input(shape=(3,))
dense1 = Dense(10,activation='relu')(inputs1)
dense1 = Dense(5,activation='relu')(dense1)
middle1 = Dense(30)(dense1)
# 분기
outputs1 = Dense(30)(middle1)
outputs1 = Dense(7)(outputs1)
outputs1 = Dense(3)(outputs1)

outputs2 = Dense(30)(middle1)
outputs2 = Dense(15)(outputs2)
outputs2 = Dense(7)(outputs2)
outputs2 = Dense(3)(outputs2)

model = Model(inputs=inputs1,outputs=[outputs1,outputs2])
model.summary()

model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x1_train,[y1_train,y2_train],epochs=200,batch_size=1,validation_split=0.2,verbose=1)

loss=model.evaluate(x1_test,[y1_test,y2_test],batch_size=1,verbose=0)



#print("model.metrics_name : ",model.metrics_names)
#print(loss)

y1_predict,y2_predict = model.predict(x1_test)
#print('===================================')
#print("y1_predict : \n",y1_predict)
#print('===================================')
#print("y2_predict : \n",y2_predict)
#print('===================================')

from sklearn.metrics import mean_squared_error,r2_score
rmse1 = mean_squared_error(y1_test,y1_predict)**0.5
rmse2 = mean_squared_error(y2_test,y2_predict)**0.5
r21 = r2_score(y1_predict,y1_test)
r22 = r2_score(y2_predict,y2_test)
print("RMSE : ",rmse1+rmse2)
print("R2   : ",(r21+r22)/2)
# y_test = np.concatenate([y1_test,y2_test])
# y_predict = np.concatenate([y1_predict,y2_predict])
# rmse = mean_squared_error(y_test,y_predict)**0.5
# r2 = r2_score(y_test,y_predict)
# print("RMSE : ",rmse)
# print("R2   : ",r2)













'''
[3.894968813256128e-06, 2.8792223361051583e-07, 3.6070473470317665e-06, 2.8792223361051583e-07, 3.6070473470317665e-06]
첫로스 +두번째 로스의 산술합       첫 모델의 로스           두번째 모델의 로스         첫번째 메트릭스          두번째 메트릭스
'''