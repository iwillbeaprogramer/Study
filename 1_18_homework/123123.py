size=20
day=2
batch_size = 4
epochs = 100000
random_state = 0
patience = 350
modelpath = "./1_18_homework/concat_17_epoch{}_batch{}_size{}_day{}_patience{}_layernormX_GRU_SAME.h5".format(epochs,batch_size,size,day,patience)

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.models import Sequential,load_model,Model
from tensorflow.keras.layers import Dense,LSTM,Conv1D,MaxPool1D,Dropout,Flatten,Input,concatenate,AveragePooling1D,BatchNormalization,GRU,LayerNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import r2_score

# 함수정의
def split_x(seq,size,col):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size),0:col].astype('float32')
        aaa.append(np.array(subset))
    return np.array(aaa)

def split_y(seq,size,day):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset = seq[i+size : i+size+day,0].astype('float32')
        aaa.append(np.array(subset))
    return np.array(aaa)

def str_to_float(input_str):
    temp = input_str
    if type(temp) == float or type(temp) == int:
        return temp
    if temp[0]!='-':
        temp = input_str.split(',')
        sum = 0
        for i in range(len(temp)):
            sum+=float(temp[-i-1])*(10**(i*3))
        return sum
    else:
        temp=temp[1:]
        temp = input_str.split(',')
        sum = 0
        for i in range(len(temp)):
            sum+=float(temp[-i-1])*(10**(i*3))
        return -sum

    
datasets_1 = np.load('./1.npy')
datasets_2 = np.load('./2.npy')
drop_col=['거래량', '금액(백만)','신용비','외국계','외인비','프로그램','외인(수량)']
col=14-len(drop_col)
scaler = MinMaxScaler()
datasets_minmaxed = scaler.fit_transform(datasets_2)
x_2 = split_x(datasets_minmaxed,size,col) 




x_1_train,x_1_test,y_1_train,y_1_test,x_2_train,x_2_test = train_test_split(x_1[:-day],y_1,x_2[:-day],test_size=0.2,random_state=random_state)
x_1_train=x_1_train.reshape(-1,size,col).astype('float32')
x_2_train=x_2_train.reshape(-1,size,col).astype('float32')
x_1_test=x_1_test.reshape(-1,size,col).astype('float32')
x_2_test=x_2_test.reshape(-1,size,col).astype('float32')


es = EarlyStopping(monitor = 'val_loss',patience=patience)
cp = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath,save_best_only=True)
inputs1 = Input(shape=(x_1_train.shape[1],x_1_train.shape[2]))
lstm1 = LSTM(64,activation='relu')(inputs1)
dense1 = Dense(1024,activation='relu')(lstm1)

inputs2 = Input(shape=(x_2_train.shape[1],x_2_train.shape[2]))
lstm2 = LSTM(64,activation='relu')(inputs2)
dense2 = Dense(1024,activation='relu')(lstm2)

model = load_model(modelpath)
y_pred = model.predict([x_1_test,x_2_test])
for i in range(1,50,7):
     print("예상_13 : {}     실제 : {}".format(round(y_pred[i][0]),y_1_test[i]))

loss = model.evaluate([x_1_test,x_2_test],y_1_test,batch_size=batch_size)
rmse = mean_squared_error(y_1_test,y_pred)**0.5
r2 = r2_score(y_1_test,y_pred)
mae = mean_absolute_error(y_1_test,y_pred)
print("loss : ",loss )
print('rmse : ',rmse,'\nr2 : ',r2,'\nmae : ',mae)

model = load_model(modelpath)
result = model.predict([x_1[-10:],x_2[-10:]])
print("1월 19일 시가 : {}".format(result[-1]))
