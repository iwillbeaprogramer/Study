#1.데이터 구성
import numpy as np

from numpy import array 

x1=array([[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[20,30],[30,40],[40,50]]) 
x2=array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],[50,60,70],[60,70,80],[70,80,90],[80,90,100],[90,100,110],[100,110,120],[2,3,4],[3,4,5],[4,5,6]])
y1=array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],[50,60,70],[60,70,80],[70,80,90],[80,90,100],[90,100,110],[100,110,120],[2,3,4],[3,4,5],[4,5,6]])
y2=array([4,5,6,7,8,9,10,11,12,13,50,60,70]) 

x1_predict=array([[55,65]]) 
x2_predict=array([[65,75,85]]) 

print("x1.shape:",x1.shape) #(13,2) #인풋딤
print("x2.shape:",x2.shape) #(13,3)
print("y1.shape:",y1.shape) #(13,3)
print("y2.shape:",y2.shape) #(13,)

#2. 모델 구성
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input

#모델1
input1=Input(shape=(2,)) #(13,2)이면 (2,)
dense1=Dense(10,activation='relu')(input1)
dense1=Dense(5, activation='relu')(dense1)

#모델2
input2=Input(shape=(3,)) #(13,3)이면 (3,)
dense2=Dense(10,activation='relu')(input2)
dense2=Dense(5, activation='relu')(dense2)
dense2=Dense(5, activation='relu')(dense2)
dense2=Dense(5, activation='relu')(dense2)

#모델병합concatenate: 사슬같이 잇다
from tensorflow.keras.layers import concatenate
merge1=concatenate([dense1,dense2])

#중간층 모델구성:생략가능
middle1=Dense(15)(merge1)
middle1=Dense(15)(middle1)

#모델 분기1
output1=Dense(30)(middle1)
output1=Dense(7)(output1)
output1=Dense(3)(output1)

#모델 분기2
output2=Dense(30)(middle1)
output2=Dense(7)(output2)
output2=Dense(7)(output2)
output2=Dense(1)(output2)

#모델 선언
model=Model(inputs=[input1,input2], #2개이상은 리스트로 묶는다.[]
            outputs=[output1,output2])

#model.summary()

#3.컴파일, 훈련
model.compile(loss='mse',optimizer='adam',metrics='mae')
model.fit([x1,x2],[y1,y2],epochs=10,verbose=1) #배치사이즈 넣으면 에러뜸,왜?

#4.평가, 예측

#.로스외의 평가지표 예측
result=model.evaluate([x1,x2],[y1,y2])  
y1_predict = model.predict([x1_predict,x2_predict])
print(result)
