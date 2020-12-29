# 실습
# x는 (100,5) 데이터 구성
# y는 (100,2) 데이터 구성
# 모델을 완성하시오
import numpy as np
x = np.array([range(100),
             range(301,401),
             range(1,101),
             range(1200,1500,3),
             range(2000,3000,10)]).T
y = np.array([range(711,811),range(201,301)]).T
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
#x_pred2=np.array([100,402,101,100,401])
#x_pred2 = x_pred2.reshape(1,5)
print(x_train.shape)
print(y_train.shape)
input1 = Input(shape=(1,5))
dense0 = Dense(100,activation='relu')(input1)
dense1 = Dense(50,activation='relu')(dense0)
dense2 = Dense(30,activation='relu')(dense1)
dense3 = Dense(20,activation='relu')(dense2)
dense4 = Dense(10,activation='relu')(dense3)
dense5 = Dense(5,activation='relu')(dense4)
outputs = Dense(2)(dense5)
model = Model(inputs = input1,outputs = outputs)
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=500,batch_size=1,validation_split=0.2,verbose=3)
loss,mae = model.evaluate(x_test,y_test,batch_size=1)
print('loss : ',loss,'\tmae : ',mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
rmse = mean_squared_error(y_test,y_predict)**0.5
r2 = r2_score(y_test,y_predict)
print('rmse : ',rmse,'\tr2 : ',r2)

for i in range(5,10):
    print('test : ',y_test[i],'predict : ',y_predict[i])








