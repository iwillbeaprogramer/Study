from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder

(x_train,y_train),(x_test,y_test) = cifar10.load_data()
es = EarlyStopping(monitor = 'val_accuracy',patience=20)
lr = ReduceLROnPlateau(monitor = 'val_accuracy',patience=7,factor = 0.6)
optimizer = Adam(lr = 0.01)

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

scaler = OneHotEncoder()
y_train = scaler.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = scaler.transform(y_test.reshape(-1,1)).toarray()


vgg16 = DenseNet121(weights='imagenet',include_top = False, input_shape=(32,32,3))
# include_top : True 면 imagenet에 들어갔던 weight를 그대로쓰는거야, False여야 우리가 원하는 사이즈로 변환해서 사용가능하다
# print(model.weights)

vgg16.trainable = False
# vgg16.summary()
# print(len(vgg16.weights))
# print(len(vgg16.trainable_weights))

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(loss = 'categorical_crossentropy',optimizer = optimizer,metrics = ['accuracy'])
model.fit(x_train,y_train,validation_split=0.2,epochs=1000,batch_size=128,callbacks=[es,lr])

result = model.evaluate(x_test,y_test)

print("loss : ",result[0],"Accuracy : ",result[1])

"""
loss :  1.1750394105911255 Accuracy :  0.6212999820709229
"""