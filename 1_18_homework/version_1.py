size=20
day=2
batch_size =64
epochs = 10000
modelpath = "./1_18_homework/concat_16_1.h5"
random_state = 0
patience = 300

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.models import Sequential,load_model,Model
from tensorflow.keras.layers import Dense,LSTM,Conv1D,MaxPool1D,Dropout,Flatten,Input,concatenate,AveragePooling1D,BatchNormalization,GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
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
        subset = seq[i:(i+size),0:col].astype('float32')
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

    
df1 = pd.read_csv('./1_18_homework/삼성전자.csv',encoding='cp949',index_col=0)
# 결측3제거
datasets_1 = df1.iloc[:662,:]
datasets_2 = df1.iloc[665:,:]
df1 = pd.concat([datasets_1,datasets_2])
df2 = pd.read_csv('./1_18_homework/삼성전자2.csv',encoding='cp949',index_col=0)
df2.drop(['전일비','Unnamed: 6'],axis='columns', inplace=True)
df2.drop(['2021-01-13'],axis=0,inplace=True)
df2.dropna(inplace=True)
datasets = pd.concat([df2,df1])
df3 = pd.read_csv('./1_18_homework/삼성전자0115.csv',encoding='cp949',index_col=0)
df3.drop(['전일비','Unnamed: 6'],axis='columns', inplace=True)
df3 = df3.iloc[0:1,:]
datasets = pd.concat([df3,datasets])

for j in [0,1,2,3,5,6,8,9,10,11,12]:
    for i in range(len(datasets.iloc[:,j])):
        datasets.iloc[i,j] = str_to_float(datasets.iloc[i,j])
datasets.iloc[662:,0:4] = datasets.iloc[662:,0:4]/50.0
datasets.iloc[662:,5] = datasets.iloc[662:,5]*50
datasets = datasets.iloc[::-1,:]
datasets.to_csv('./1_18_homework/temp_file.csv')
datasets = pd.read_csv('./1_18_homework/temp_file.csv',index_col=0)
datasets_1 = datasets.iloc[1314:,:]
datasets_1["고가-저가"]=datasets_1['고가']-datasets_1['저가']
drop_col=['거래량', '금액(백만)','신용비','외국계','외인비','프로그램','외인(수량)']
datasets_1.drop(drop_col, axis='columns', inplace=True)
col=14-len(drop_col)
y_1 = datasets_1.iloc[size+1:,0]
scaler = MinMaxScaler()
datasets_minmaxed = scaler.fit_transform(datasets_1)
x_1 = split_x(datasets_minmaxed,size,col)

df = pd.read_csv('./1_18_homework/KODEX 코스닥150 선물인버스.csv',encoding='cp949',index_col=0)
dfdf1 = df.iloc[:421,:]
dfdf2 = df.iloc[424:,:]
df = pd.concat([dfdf1,dfdf2])

df.drop(['전일비','Unnamed: 6'],axis='columns', inplace=True)
drop_col=['거래량', '금액(백만)','신용비','외국계','외인비','프로그램','외인(수량)']
df.drop(drop_col, axis='columns', inplace=True)
for j in [0,1,2,3,5,6]:
    for i in range(len(df.iloc[:,j])):
        df.iloc[i,j] = str_to_float(df.iloc[i,j])
df.iloc[662:,0:4] = df.iloc[662:,0:4]/50.0
df.iloc[662:,5] = df.iloc[662:,5]*50
df = df.iloc[::-1,:]
df["고가-저가"]=df['고가']-df['저가']
df.to_csv('./1_18_homework/temp_file_2.csv')
datasets_2 = pd.read_csv('./1_18_homework/temp_file_2.csv',index_col=0)
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

concat = concatenate([dense1,dense2])
dense = Dense(512,activation='relu')(concat)
dense = Dense(256,activation='relu')(dense)
dense = Dense(128,activation='relu')(dense)
dense = Dense(64,activation='relu')(dense)
dense = Dense(32,activation='relu')(dense)
dense = Dense(16,activation='relu')(dense)
dense = Dense(8,activation='relu')(dense)
outputs = Dense(1)(dense)
model = Model(inputs=[inputs1,inputs2],outputs=outputs)
model.compile(loss = 'mse',optimizer = 'adam',metrics=['mae'])
hist = model.fit([x_1_train,x_2_train],y_1_train,validation_split=0.2,epochs=epochs,batch_size=batch_size,verbose=1,callbacks=[es,cp])

y_pred = model.predict([x_1_test,x_2_test])
for i in range(1,200,7):
     print("예상_13 : {}     실제 : {}".format(round(y_pred[i][0]),y_1_test[i]))

loss = model.evaluate([x_1_test,x_2_test],y_1_test,batch_size=batch_size)
rmse = mean_squared_error(y_1_test,y_pred)**0.5
r2 = r2_score(y_1_test,y_pred)
print("loss : ",loss )
print('rmse : ',rmse,'\tr2 : ',r2)