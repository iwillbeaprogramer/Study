import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.models import Sequential,load_model,Model
from tensorflow.keras.layers import Dense,LSTM,Conv1D,MaxPool1D,Dropout,Flatten,Input,concatenate,AveragePooling1D,BatchNormalization
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

def str_to_float(input_str):
    temp = input_str
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


datasets = pd.read_csv("./samsung/삼성전자.csv",encoding='cp949',index_col=0)
#전처리 
#1-1 분할(결측치가 있는 3개의행 제거)
datasets_1 = datasets.iloc[:662,:]
datasets_2 = datasets.iloc[665:,:]

datasets = pd.concat([datasets_1,datasets_2])

# 50으로 나누고 곱함
# str -> florat


#필요한 열에 타입변환 및 순서 바꿈
for j in [0,1,2,3,5,6,8,9,10,11,12]:
    for i in range(len(datasets.iloc[:,j])):
        datasets.iloc[i,j] = str_to_float(datasets.iloc[i,j])
datasets.iloc[662:,0:4] = datasets.iloc[662:,0:4]/50.0
datasets.iloc[662:,5] = datasets.iloc[662:,5]*50
datasets = datasets.iloc[::-1,:]
datasets.to_csv('./samsung/csv_ogirinal_version2.csv')
datasets = pd.read_csv('./samsung/csv_ogirinal_version2.csv',index_col=0)
np.save("./samsung/삼성전자_ogirinal_version2.npy",arr=datasets)





#열추가
datasets["고가-저가"]=datasets['고가']-datasets['저가']

#y데이터 생성
#[시가,고가,저가,종가,등락률,거래량,금액(백만),신용비,개인,기관,외인(수량),외국계,프로그램,외인비,고-저]
#[   0   1    2    3    4     5      6          7    8    9         10       11    12     13     14]
print(datasets)

#열제거
drop_col=['거래량', '금액(백만)','신용비','외국계','외인비','프로그램','외인(수량)']
datasets.drop(drop_col, axis='columns', inplace=True)

size=20
col=14-len(drop_col)
print(datasets)


























y = datasets.iloc[size-1:,1].values #(2378,)


#MinMax
scaler = MinMaxScaler()
datasets_minmaxed = scaler.fit_transform(datasets)

#train_test_split
x = split_x(datasets_minmaxed,size,col) # (2378,20,14)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2)
x_train=x_train.reshape(-1,size,col).astype('float32')
x_test=x_test.reshape(-1,size,col).astype('float32')
x_val=x_val.reshape(-1,size,col).astype('float32')
'''
modelpath = "./samsung//Samsung_best_model_s_col{}_original_version2_2.h5".format(col)
es = EarlyStopping(monitor = 'val_loss',patience=200)
cp = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath,save_best_only=True)
model = Sequential()
model.add(LSTM(64,activation='relu',input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dense(1024,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))
model.compile(loss = 'mse',optimizer = 'adam')
hist = model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=10000,batch_size=4,verbose=1,callbacks=[es,cp])
'''
model = load_model("./samsung/Samsung_best_model_s_col7_original_version2_2.h5")
y_pred = model.predict(x_test)

for i in range(1,200,7):
    print("예상_13 : {}     실제 : {}".format(round(y_pred[i][0]),y_test[i]))

loss = model.evaluate(x_test,y_test,batch_size=8)
print("loss : ",loss )

y_pred = model.predict(x_test)
rmse = mean_squared_error(y_test,y_pred)**0.5
r2 = r2_score(y_test,y_pred)

print('rmse : ',rmse,'\tr2 : ',r2)

