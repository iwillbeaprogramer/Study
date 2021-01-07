from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss',patience=20)

datasets = load_wine()
x = datasets.data
y = datasets.target

encoder = OneHotEncoder()
y = encoder.fit_transform(y.reshape(-1,1)).toarray()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
x_val = scaler.fit_transform(x_val)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)

model = Sequential()
model.add(LSTM(128,activation='relu',input_shape = (x_train.shape[1],x_train.shape[2])))
model.add(Dense(128,activation='relu',input_dim=13))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(3,activation='softmax'))
model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=300,batch_size=4,callbacks=[early_stopping])
loss = model.evaluate(x_test,y_test,batch_size=4)
y_pred = model.predict(x_test)

print('loss : ',loss[0],'\naccuracy : ',loss[1])

'''
DNN
loss :  3.391478821868077e-05 
accuracy :  1.0

LSTM
loss :  0.30595341324806213 
accuracy :  0.9166666865348816
'''