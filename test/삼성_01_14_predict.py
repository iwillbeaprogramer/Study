import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,LSTM,Conv1D,MaxPool1D,Dropout,Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential,load_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score,mean_squared_error

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

cols = ['시가','고가','저가','종가','등락률','거래량','금액(백만)','신용비','개인','기관','외인(수량)','외국계','프로그램','외인비']
arr = np.load('./test/01_14.npy')
datasets = pd.DataFrame(arr,columns=cols)

datasets["고가-저가"]=datasets['고가']-datasets['저가']
drop_col=['거래량', '금액(백만)','신용비','외국계','외인비','프로그램','외인(수량)']
datasets.drop(drop_col, axis='columns', inplace=True)
size=20
col=14-len(drop_col)
y = datasets.iloc[size-1:,1].values 
scaler = MinMaxScaler()
datasets_minmaxed = scaler.fit_transform(datasets)
x = split_x(datasets_minmaxed,size,col) 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2)
x_train=x_train.reshape(-1,size,col).astype('float32')
x_test=x_test.reshape(-1,size,col).astype('float32')
x_val=x_val.reshape(-1,size,col).astype('float32')
model_14 = load_model('./test/1_14_best_model.h5')

model_14 = load_model('./test/1_14_best_model.h5')
y_pred = model_14.predict(x_test)
jen_14 = model_14.predict(x[-20:])

for i in range(1,200,7):
    print("예상 : {}     실제 : {}".format(round(y_pred[i][0]),y_test[i]))

'''

datasets = pd.DataFrame(arr,columns=cols)
datasets.drop(['거래량', '금액(백만)','신용비','외국계','프로그램'], axis='columns', inplace=True)
size=20
col=6
y = datasets.iloc[size-1:,3].values 
scaler = MinMaxScaler()
datasets_minmaxed = scaler.fit_transform(datasets)
x = split_x(datasets_minmaxed,size,col)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2)
x_train=x_train.reshape(-1,size,col).astype('float32')
x_test=x_test.reshape(-1,size,col).astype('float32')
x_val=x_val.reshape(-1,size,col).astype('float32')


#model_14 = load_model('./test/1_14_best_model.h5')
model_13 = load_model('./test/1_13_best_model.h5')
y_pred_13 = model_13.predict(x_test)
#y_pred_14 = model_14.predict(x_test)
for i in range(1,100,5):
    print("예상_13 : {}     예상_14 : {}     실제 : {}".format(round(y_pred_13[i][0]),round(y_pred_14[i][0]),y_test[i]))


jen_13 = model_13.predict(x[-20:])
#jen_14 = model_14.predict(x[-20:])
print(jen_13[-1][0],jen_14[-1][0])
tomorrow = (jen_13[-1][0]+jen_14[-1][0])/2



rmse_13 = mean_squared_error(y_test,y_pred_13)**0.5
r2_13 = r2_score(y_test,y_pred_13)

rmse_14 = mean_squared_error(y_test,y_pred_14)**0.5
r2_14 = r2_score(y_test,y_pred_14)
print("\n13\n")
print('rmse : ',rmse_13,'\tr2 : ',r2_13)
print("Tomorrow : : ",round(tomorrow,-2))

print("\n14\n")
print('rmse : ',rmse_14,'\tr2 : ',r2_14)
print("Tomorrow : : ",round(tomorrow,-2))
'''