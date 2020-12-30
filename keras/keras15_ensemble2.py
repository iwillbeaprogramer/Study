# 실습 디:1 잉싱블을 구현하시오
import numpy as np
x1 = np.array([range(100),range(301,401),range(1,101)]).T
x2 = np.array([range(101,201),range(411,511),range(100,200)]).T

y1 = np.array([range(711,811),range(1,101),range(201,301)]).T

# y2 = np.array([range(501,601),range(711,811),range(100)]).T

from sklearn.model_selection import train_test_split
x1_train,x1_test,y1_train,y1_test = train_test_split(x1,y1,test_size=0.2,shuffle=False)
#x2_train,x2_test,y2_train,y2_test = train_test_split(x2,x2,test_size=0.2,shuffle=False)
x2_train,x2_test = train_test_split(x2,test_size=0.2,shuffle=False)
'''
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(x1, x2, y1, train_size = 0.8, shuffle = False)
'''


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input,Dense,LSTM,Conv2D,Concatenate,concatenate

inputs1 = Input(shape=(3,))
dense1 = Dense(10,activation='relu')(inputs1)
dense1 = Dense(5,activation='relu')(dense1)
#outputs1 = Dense(3)(dense1)

inputs2 = Input(shape=(3,))
dense2 = Dense(10,activation='relu')(inputs2)
dense2 = Dense(5,activation='relu')(dense2)
dense2 = Dense(5,activation='relu')(dense2)
dense2 = Dense(5,activation='relu')(dense2)
#outputs2 = Dense(3)(dense2)
merge1 = concatenate([dense1,dense2])
middle1 = Dense(30)(merge1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)
middle1 = Dense(5)(middle1)
outputs = Dense(3)(middle1)

model = Model(inputs=[inputs1,inputs2],outputs=outputs)
model.summary()

model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit([x1_train,x2_train],y1_train,epochs=300,batch_size=1,validation_split=0.2,verbose=1)
loss=model.evaluate([x1_test,x2_test],y1_test,batch_size=1,verbose=0)

print("model.metrics_name : ",model.metrics_names)
print(loss)

y1_predict = model.predict([x1_test,x2_test])
#print('===================================')
#print("y1_predict : \n",y1_predict)
#print('===================================')
#print("y2_predict : \n",y2_predict)
#print('===================================')

from sklearn.metrics import mean_squared_error,r2_score
rmse1 = mean_squared_error(y1_test,y1_predict)**0.5
r21 = r2_score(y1_predict,y1_test)
print("RMSE : ",rmse1)
print("R2   : ",(r21))

