# 나를 찍어서 내가 남자인지 여자인지에 대해 결과

from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt

model = load_model("../data/image/male_female/models/best.h5")
img = cv2.imread("../data/image/male_female/my_picture.jpg",cv2.COLOR_BGR2RGB)
img = cv2.resize(img,dsize=(100,100))/255.0
plt.imshow(img)
plt.show()
result = model.predict(np.array([img]))
# print(result)
def gender(value):
    if round(value)==1:
        result = '남자'
    else:
        result = '여자'
    return result

print(100*result[0][0],"% 확률로 {}입니다.".format(gender(result[0][0])))
print("Accuracy = ",result[0][0])