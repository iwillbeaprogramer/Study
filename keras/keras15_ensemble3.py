# 다:다 앙상블을 구현하시오.
import numpy as np
x1 = np.array([range(100),range(301,401),range(1,101)]).T
y1 = np.array([range(711,811),range(1,101),range(201,301)]).T
x2 = np.array([range(101,201),range(411,511),range(100,200)]).T
y2 = np.array([range(501,601),range(711,811),range(100)]).T
y3 = np.array([range(601,701),range(811,911),range(1100,1200)]).T

from sklearn.model_selection import train_test_split
x1_train,x1_test,y1_train,y1_test = train_test_split(x1,y1,test_size=0.2,shuffle=False)
x2_train,x2_test,y2_train,y2_test, y3_train, y3_test= train_test_split(x2,y2,y3,test_size=0.2,shuffle=False)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input,Dense,LSTM,Conv2D,Concatenate,concatenate

inputs1 = Input(shape=(3,))
dense1 = Dense(10,activation='relu')(inputs1)
dense1 = Dense(5,activation='relu')(dense1)
#outputs1 = Dense(3)(dense1)

inputs2 = Input(shape=(3,))
dense2 = Dense(10,activation='relu')(inputs2)
dense2 = Dense(5,activation='relu')(dense2)
#outputs2 = Dense(3)(dense2)
merge1 = concatenate([dense1,dense2])
middle1 = Dense(30)(merge1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)
# 분기
outputs1 = Dense(30)(middle1)
outputs1 = Dense(7)(outputs1)
outputs1 = Dense(3)(outputs1)

outputs2 = Dense(30)(middle1)
outputs2 = Dense(15)(outputs2)
outputs2 = Dense(7)(outputs2)
outputs2 = Dense(3)(outputs2)

outputs3 = Dense(40)(middle1)
outputs3 = Dense(20)(outputs3)
outputs3 = Dense(10)(outputs3)
outputs3 = Dense(3)(outputs3)

model = Model(inputs=[inputs1,inputs2],outputs=[outputs1,outputs2,outputs3])
model.summary()

model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit([x1_train,x2_train],[y1_train,y2_train,y3_train],epochs=300,batch_size=1,validation_split=0.2,verbose=1)

loss=model.evaluate([x1_test,x2_test],[y1_test,y2_test,y2_test],batch_size=1,verbose=0)



#print("model.metrics_name : ",model.metrics_names)
#print(loss)

y1_predict,y2_predict,y3_predict = model.predict([x1_test,x2_test])
a,b = np.array([100,401,101]).reshape(1,3),np.array([201,511,200]).reshape(1,3)
#print('===================================')
#print("y1_predict : \n",y1_predict)
#print('===================================')
#print("y2_predict : \n",y2_predict)
#print('===================================')

from sklearn.metrics import mean_squared_error,r2_score
rmse1 = mean_squared_error(y1_test,y1_predict)**0.5
rmse2 = mean_squared_error(y2_test,y2_predict)**0.5
rmse3 = mean_squared_error(y3_test,y2_predict)**0.5
r21 = r2_score(y1_predict,y1_test)
r22 = r2_score(y2_predict,y2_test)
r23 = r2_score(y3_predict,y3_test)
print("RMSE : ",(rmse1+rmse2+rmse3)/3)
print("R2   : ",(r21+r22+r23)/3)
# y_test = np.concatenate([y1_test,y2_test])
# y_predict = np.concatenate([y1_predict,y2_predict])
# rmse = mean_squared_error(y_test,y_predict)**0.5
# r2 = r2_score(y_test,y_predict)
# print("RMSE : ",rmse)
# print("R2   : ",r2)
print(model.predict([a,b]))
print("811 101 301 601 811 100 701 911 1200")













'''
[3.894968813256128e-06, 2.8792223361051583e-07, 3.6070473470317665e-06, 2.8792223361051583e-07, 3.6070473470317665e-06]
첫로스 +두번째 로스의 산술합       첫 모델의 로스           두번째 모델의 로스         첫번째 메트릭스          두번째 메트릭스
'''