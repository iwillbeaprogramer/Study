# 이미지는 
# 개 고양이 라이언 슈트
# 파일명:
# dog1.jpg,cat1.jpg.......

import numpy as np
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16

img_dog = load_img("../data/image/VGG/dog1.jpg",target_size=(224,224))
img_cat = load_img("../data/image/VGG/cat1.jpg",target_size=(224,224))
img_lyon = load_img("../data/image/VGG/lyon1.jpg",target_size=(224,224))
img_suit = load_img("../data/image/VGG/suit1.jpg",target_size=(224,224))


arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_lyon = img_to_array(img_lyon)
arr_suit = img_to_array(img_suit)

arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_lyon = preprocess_input(arr_lyon)
arr_suit = preprocess_input(arr_suit)

print(arr_dog.shape)
arr_inputs = np.array([arr_dog,arr_cat,arr_lyon,arr_suit])
# 케라스로 당겨오면 RGB 형식이다   // VGG는 BGR형식의 인풋을 받는다

model = VGG16()
results = model.predict(arr_inputs)
# print(results)
print('result.shape : ',results.shape)

# 이미지 결과 확인

from tensorflow.keras.applications.vgg16 import decode_predictions

print("====================================================")
print("result 0 : ",decode_predictions(results)[0])
print("====================================================")
print("result 1 : ",decode_predictions(results)[1])
print("====================================================")
print("result 2 : ",decode_predictions(results)[2])
print("====================================================")
print("result 3 : ",decode_predictions(results)[3])
print("====================================================")



