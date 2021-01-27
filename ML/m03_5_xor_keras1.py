from sklearn.svm import LinearSVC,SVC
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x_data = np.array([[0,0],[0,1],[1,0],[1,1]])
y_data = np.array([0,1,1,0])

model = Sequential()
model.add(Dense(1,input_dim=2,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_data,y_data,epochs = 20,batch_size=1)

y_pred = model.predict(x_data)
result = model.evaluate(x_data,y_data)
print(x_data,"의 예측결과 : ",y_pred)
print('model.score : ',result[1])
#acc = accuracy_score(y_data,y_pred)
print("acc : ",result[1])

