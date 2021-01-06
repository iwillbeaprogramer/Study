#1. 데이터
import numpy as np
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = np.array([4,5,6,7])

# 한개식 잘라서 작업하기 위해 x의 shape를 바꿔야함
x = x.reshape(4,3,1) #  = np.array([[[1],[2],[3]],[[2],[3],[4]],[[3],[4],[5]],[[4],[5],[6]]])


#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(10, activation='relu',return_sequences=True,input_shape = (3,1))) #LSTM레이어를 쓰려면 데이터구조가 3차원이여야함 / 한개식 잘라서 작업을함 여기서는 (4,3)-> (4,3,1) LSTM에 넣기 위해서
model.add(LSTM(10))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
model.summary()


'''
# 3. 모델 컴파일 훈련
model.compile(loss = 'mse',optimizer='adam')
model.fit(x,y,epochs=100,batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x,y,batch_size=1)
print("loss : ",loss)
x_pred = np.array([5,6,7]) # (3,) ->(1,3,1)
x_pred = x_pred.reshape(1,3,1)
y_pred = model.predict(x_pred)
print('y_pred : ',y_pred)




# LSTM 행 렬 몇개식 짜르는지
# LSTM 행 렬 몇개식 짜르는지
# LSTM 행 렬 몇개식 짜르는지
# LSTM 행 렬 몇개식 짜르는지
# LSTM 행 렬 몇개식 짜르는지
# LSTM 행 렬 몇개식 짜르는지
# LSTM 행 렬 몇개식 짜르는지
# LSTM 행 렬 몇개식 짜르는지
# LSTM 행 렬 몇개식 짜르는지
# LSTM 행 렬 몇개식 짜르는지
# LSTM 행 렬 몇개식 짜르는지
# LSTM 행 렬 몇개식 짜르는지
# LSTM 행 렬 몇개식 짜르는지
# LSTM 행 렬 몇개식 짜르는지
# LSTM 행 렬 몇개식 짜르는지
# LSTM 행 렬 몇개식 짜르는지
'''
'''
4x(1+1+10)x10
첫번쨰 1은 인풋딤 두번째일은 바이어스 3번쨰는 10개가 돌아오는거
(행,열,몇개식 짜르는지)
(배치사이즈,타임스탭,인풋딤)
lstm에 인풋쉐이프 에서는  input_shape = (time_step, input_dim) 이렇게 넣어주거나
좀 훈련속도를 늘리기 위해 배치까지 정하려면 batch_input_shape = (batct_size,time_stemp,input_dim)

LSTM의 액티베이션 디폴트? 하이퍼볼리기 탄젠트
LSTM 게이트 : output_gate,input_gate,forget_gate,memory_cell?
'''

'''
input_shape가 (none,100,5) ->(none,100,10)

일단 return_sequences=True 이걸 외우자
'''








































'''
삼성전자의 주가 예시
1/1 8.0
1/2 7.9
1/3 8.2
1/4 8.5
1/5 8.4
1/6 ?

1,2,3일을 이용해서 4일을 예측
2,3,4일을 이용해서 5일을 예측
3,4,5일을 이용해서 6일을 예측
연산을 하나식 시키겠다? -> (3,1) -> (3개를묶어서,1개식)




1/1 75
1/2 74 
1/3 80
1/4 77
1/5 73 
1/6 72
1/7 71
1/8 ?
이걸 3일치식 끊어서 맞춘다
75 74 80        77
74 80 77        73
80 77 73        72
77 73 72        71

shape :  (4,3) -> (4,3,1)
'''