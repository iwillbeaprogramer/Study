import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

datasets = load_breast_cancer()
print(datasets.feature_names)
print(datasets.DESCR)

x = datasets.data
y = to_categorical(datasets.target)
print(x.shape) # (569,30)
print(y.shape) # (569,)
print(y[0:100])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

early_stopping = EarlyStopping(monitor='val_loss',patience=25,mode='auto')

model = Sequential()
model.add(Dense(1024,activation='relu',input_shape=(30,)))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(2,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
hist = model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=20000,verbose=1,callbacks=[early_stopping],batch_size=2)

loss = model.evaluate(x_test,y_test,batch_size=1)
y_prediction = np.argmax(model.predict(x_test),axis=-1)

'''
count=0
for i in range(len(y_prediction)):
    print('실제값 : {} , 예측값 : {}'.format(y_test[i].tolist().index(1),y_prediction[i]))
    if y_test[i].tolist().index(1) !=y_prediction[i]:
        count+=1
'''
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epochs')
plt.legend(['train_loss','val_loss','train_acc','val_acc'])
plt.show()


#print("틀린갯수 : ",count)
print('loss : ',loss[1])

'''
accuracy :  0.9912280440330505
'''



