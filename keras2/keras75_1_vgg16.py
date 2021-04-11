from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Sequential

model = VGG16(weights='imagenet',include_top = False,input_shape=(32,32,3))
# include_top : True 면 imagenet에 들어갔던 weight를 그대로쓰는거야, False여야 우리가 원하는 사이즈로 변환해서 사용가능하다
# print(model.weights)


model.trainable = False
model.summary()
print(len(model.weights))
print(len(model.trainable_weights))

model.trainable = True
model.summary()
print(len(model.weights))
print(len(model.trainable_weights))