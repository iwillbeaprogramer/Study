import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

modelpath = '../data/modelCheckpoint/k46_fashion_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath = modelpath, monitor='val_loss',save_best_only = True, mode = 'auto')
es = EarlyStopping(monitor='loss',patience = 10)
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()
#  (60000, 28, 28) (60000,)
#  (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255
x_test= x_test.reshape(10000,28,28,1).astype('float32')/255

onehot = OneHotEncoder()
y_train = onehot.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = onehot.transform(y_test.reshape(-1,1)).toarray()
print(y_test.shape)
model = Sequential()
model.add(Conv2D(filters = 256,kernel_size = 2, strides=1 , input_shape=(28,28,1),padding='valid'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics = ['accuracy'])
model.fit(x_train,y_train,validation_split=0.2,epochs=200,callbacks=[es,cp],batch_size=4,verbose=1)

loss = model.evaluate(x_test,y_test,batch_size=4)
y_predict = model.predict(x_test)
y_test = np.argmax(y_test,axis=-1)
y_predict = np.argmax(y_predict,axis=-1)

for i in range(10,20):
    print('예상 : {}    실제 : {}'.format(y_predict[i],y_test[i]))

print('loss : {}    accuracy : {}'.format(loss[0],loss[1]))


'''
CNN
loss : 1.019891619682312    accuracy : 0.8964999914169312
mc loss : 0.9870533347129822    accuracy : 0.8888999819755554
DNN
loss : 1.501900315284729    accuracy : 0.8668000102043152

LSTM
loss : 2.3025221824645996    accuracy : 0.10000000149011612
'''