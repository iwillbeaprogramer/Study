import numpy as np
import pandas as pd
from sklearn.datasets import load_boston

dataset = load_boston()
x, y = dataset.data,dataset.target
print(x.shape) # (506,13)
print(y.shape) # (506,)
print('=======================')
print(x[0:5])
print(y[:10])
print(np.max(x),np.min(x))
print(dataset.feature_names)
print(dataset.DESCR)

from sklearn.preprocessing import StandardScaler
print(np.max(x),np.min(x))



from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)


from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Dense(128,input_shape=(13,),activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,validation_split=0.2,epochs=200,batch_size=1,verbose=1)

loss = model.evaluate(x_test,y_test,batch_size=1,verbose=0)
print('loss : ',loss)
y_predict = model.predict(x_test)
rmse = mean_squared_error(y_predict,y_test)
r2 = r2_score(y_predict,y_test)
print('rmse : ',rmse)
print('r2_Score : ',r2)


'''
model.add(Dense(128,input_shape=(13,),activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))


x전체 한번에 전처리
loss :  24.84088897705078
rmse :  24.84089280428972
r2_Score :  0.6659843508169363

loss :  25.806140899658203
rmse :  25.806147136291415
r2_Score :  0.5949845823230864

'''