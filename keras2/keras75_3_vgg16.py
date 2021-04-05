from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Sequential

vgg16 = VGG16(weights='imagenet',include_top = False,input_shape=(32,32,3))
# include_top : True 면 imagenet에 들어갔던 weight를 그대로쓰는거야, False여야 우리가 원하는 사이즈로 변환해서 사용가능하다
# print(model.weights)

vgg16.trainable = False
vgg16.summary()
print(len(vgg16.weights))
print(len(vgg16.trainable_weights))

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))
model.summary()

print("그냥 가중치의 수 : ", len(model.weights))
print("동결하기 전 훈련되는 가중치의 수 : ", len(model.trainable_weights))

########################여기중요

import pandas as pd
pd.set_option("max_colwidth",-1)
layers =[(layer,layer.name,layer.trainable) for layer in model.layers]
aaa = pd.DataFrame(layers,columns = ["Layer Type","Layer Name",'Layer Trainable'])

print(aaa)