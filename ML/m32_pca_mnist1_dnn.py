# m31로 만든 0.95 이상의 n_component = ? 를 사용해 dnn모델을 만들것import numpy as np
# mnist dnn보다 성능 좋게 만들어라!!
# cnn과 비교
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test) = mnist.load_data()

pca = PCA()
x = np.append(x_train,x_test,axis=0)
x = pca.fit_transform(x.reshape(-1,28*28))/255.0
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum>=0.95)+1

x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.15)


pca = PCA(n_components=d)
x_train = pca.fit_transform(x_train.reshape(-1,784)/255.0)
x_test = pca.transform(x_test.reshape(-1,784)/255.0)
x_val = pca.transform(x_val.reshape(-1,784)/255.0)

onehot = OneHotEncoder()
y_train = onehot.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = onehot.transform(y_test.reshape(-1,1)).toarray()
y_val = onehot.transform(y_val.reshape(-1,1)).toarray()
es = EarlyStopping(monitor = 'val_loss',mode='auto',patience=10)


model = Sequential()
model.add(Dense(2*d,input_dim=d,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics=['accuracy'])
model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=100,verbose=1,callbacks=[es],batch_size=16)

result = model.evaluate(x_test,y_test)
print('result : ',result)

'''
result :  [0.1430211365222931, 0.9789000153541565]
'''








