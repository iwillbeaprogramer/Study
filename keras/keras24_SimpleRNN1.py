#1. 데이터
import numpy as np
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = np.array([4,5,6,7])

# 한개식 잘라서 작업하기 위해 x의 shape를 바꿔야함
x = x.reshape(4,3,1) #  = np.array([[[1],[2],[3]],[[2],[3],[4]],[[3],[4],[5]],[[4],[5],[6]]])


#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN,LSTM
model = Sequential()
model.add(SimpleRNN(10, activation='relu',input_shape = (3,1))) #LSTM레이어를 쓰려면 데이터구조가 3차원이여야함 / 한개식 잘라서 작업을함 여기서는 (4,3)-> (4,3,1) LSTM에 넣기 위해서
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
model.summary()



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


'''
LSTM
loss :  0.06028993800282478
y_pred :  [[8.413827]]


loss :  0.00016063085058704019
y_pred :  [[8.242601]]
'''

'''
SimpleRNN 의 파라미터는
NODE*(NODE+input_dim+1)
ACTIVATION tanh
LSTM은 4배
'''