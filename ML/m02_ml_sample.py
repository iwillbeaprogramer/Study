import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,LabelEncoder
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
# from keras.utils.np_utils import to_categorical
from sklearn.svm import LinearSVC
datasets = load_iris()
x = datasets.data
y = datasets.target
# x,y = load_iris(return_X_y=True)
#onehot = OneHotEncoder()
#onehot.fit(y.reshape(-1,1))
#y = onehot.transform(y.reshape(-1,1)).toarray()



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

early_stopping = EarlyStopping(monitor='val_loss',patience=30,mode='auto')

# model = Sequential()
# model.add(Dense(1024,activation='relu',input_shape=(4,)))
# model.add(Dense(512,activation='relu'))
# model.add(Dense(256,activation='relu'))
# model.add(Dense(128,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(32,activation='relu'))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(8,activation='relu'))
# model.add(Dense(3,activation='softmax'))
# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=20000,verbose=1,callbacks=[early_stopping],batch_size=1)

# loss = model.evaluate(x_test,y_test,batch_size=1)
# y_prediction = np.argmax(model.predict(x_test),axis=-1)
model = LinearSVC()
model.fit(x_train,y_train)
result = model.score(x_test,y_test)
print(result)
y_pred = model.predict(x_test[-5:-1])
print(y_pred)
print(y_test[-5:-1])


'''
accuracy :  1.0
'''



















'''
# 내 코드
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2,stratify = y_train)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)




early_stopping = EarlyStopping(monitor='val_loss',patience=25,mode='auto')
model = Sequential()
model.add(Dense(32,activation='relu',input_shape=(4,)))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(3,activation = 'softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=20000,verbose=1,callbacks=[early_stopping],batch_size=2)


loss = model.evaluate(x_test,y_test,batch_size=2)
y_prediction = model.predict(x_test)
rmse = mean_squared_error(y_prediction,y_test)**0.5
print('accuracy : ',loss[1])

for i in range(0,5):
    label = y_test[i]
    prediction = y_prediction[i]
    print("실제 : {}, 예측 : {}".format(label,prediction))
'''