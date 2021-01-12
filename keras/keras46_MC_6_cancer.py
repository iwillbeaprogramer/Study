import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape) # (569,30)
print(y.shape) # (569,)
print(y[0:100])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2,shuffle=True)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

modelpath = '../data/modelCheckpoint/k46_cancer_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(monitor='val_loss',filepath=modelpath,save_best_only=True)
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
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=10000,verbose=1,callbacks=[early_stopping,cp],batch_size=2)

loss = model.evaluate(x_test,y_test,batch_size=2)
y_prediction = model.predict_classes(x_test)

count=0
for i in range(len(y_test)):
    print('실제 : {} , 예측 : {}'.format(y_test[i],y_prediction[i][0]))
    if y_test[i] != y_prediction[i][0]:
        count+=1
        print('여기다')
print('loss : ',loss[1])

'''
accuracy :  0.9824561476707458
'''
