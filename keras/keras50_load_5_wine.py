import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt

modelpath = '../data/modelCheckpoint/k46_wine_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss',patience=10)

x = np.load('../data/npy/wine_x.npy')
y = np.load('../data/npy/wine_y.npy')

encoder = OneHotEncoder()
y = encoder.fit_transform(y.reshape(-1,1)).toarray()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
x_val = scaler.fit_transform(x_val)

model = Sequential()
model.add(Dense(128,activation='relu',input_dim=13))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(3,activation='softmax'))
model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
hist = model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=200,batch_size=4,callbacks=[early_stopping,cp])
loss = model.evaluate(x_test,y_test,batch_size=4)
y_pred = model.predict(x_test)

print('loss : ',loss[0],'\naccuracy : ',loss[1])

'''
DNN
loss :  3.391478821868077e-05 
accuracy :  1.0

'''
plt.rc('font', family='Malgun Gothic')
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('왓 & acc')
plt.ylabel('loss, acc')
plt.xlabel('epochs')
plt.legend(['train_loss','val_loss','train_acc','val_acc'])
plt.show()