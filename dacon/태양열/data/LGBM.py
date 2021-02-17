###트레인 7일치씩 해서 다음 2일치를 예측하기
## cpu로 돌아간다.

import numpy as np
import pandas as pd
import os
import glob
import random
import warnings

warnings.filterwarnings("ignore")

train= pd.read_csv('./데이콘/태양열/data/train/train.csv', encoding='cp949')
# print(train.head())
#337 7일차 337행
# print(train.shape) # (52560,8)

submission = pd.read_csv('./데이콘/태양열/data/sample_submission.csv', encoding='cp949')
# print(submission.tail())


def preprocess_data(data, is_train=True):
    
    temp = data.copy()
    temp['Hour'] = data['Hour']*60+data['Minute']
    sunup=[]
    for i in range(len(temp)):
        if temp.iloc[i,8]>0:
            sunup.append(1)
        else:
            sunup.append(0)
    temp['Sun_up'] = sunup
    temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T','Sun_up']]

    if is_train==True:          
    
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill')
        temp = temp.dropna()
        
        return temp.iloc[:-96]

    elif is_train==False:
        
        temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T','Sun_up']]
                              
        return temp.iloc[-48:, :]


df_train = preprocess_data(train)
df_train.iloc[:48]
# print(df_train.head())

train.iloc[48:96]
train.iloc[48+48:96+48]

# print(df_train.tail())

df_test = []

for i in range(81):
    file_path = './데이콘/태양열/data/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_test.append(temp)

x_test = pd.concat(df_test)
# print(x_test.shape) #(3888, 7)

# print(x_test.head(48))
# print(df_train.head())
df_train.iloc[-48:]

from sklearn.model_selection import train_test_split
x_train_1, x_val_1, y_train_1, y_val_1 = train_test_split(
    df_train.iloc[:, :-2], df_train.iloc[:, -2], test_size=0.3, random_state=0)
x_train_2, x_val_2, y_train_2, y_val_2 = train_test_split(
    df_train.iloc[:, :-2], df_train.iloc[:, -1], test_size=0.3, random_state=0)

# print(x_train_1.head())
# print(x_test.head())

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# print(x_train_1.shape)
# print(x_val_1.shape)
# print(y_train_1.shape)
# print(y_val_1.shape)
# print(x_train_2.shape)
# print(x_val_2.shape)
# print(y_train_2.shape)
# print(y_val_2.shape)

###########LGBM
from lightgbm import LGBMRegressor

# Get the model and the predictions in (a) - (b)
def LGBM(q, x_train, y_train, x_valid, y_valid, x_test):
    
    # (a) Modeling  
    model = LGBMRegressor(objective='quantile', alpha=q,
                         n_estimators=1000000, bagging_fraction=0.7, learning_rate=0.027, subsample=0.7, num_iteration=1000)                   
                         
                         
    model.fit(x_train, y_train, eval_metric = ['quantile'], 
          eval_set=[(x_valid, y_valid)], early_stopping_rounds=300, verbose=500)

    # (b) Predictions
    pred = pd.Series(model.predict(x_test).round(2)) #소수점 2번째 자리까지
    return pred, model


# Target 예측
def train_data(x_train, y_train, x_valid, y_valid, x_test):

    LGBM_models=[]
    LGBM_actual_pred = pd.DataFrame()

    for q in quantiles:
        print(q)
        pred , model = LGBM(q, x_train, y_train, x_valid, y_valid, x_test)
        LGBM_models.append(model)
        LGBM_actual_pred = pd.concat([LGBM_actual_pred,pred],axis=1)

    LGBM_actual_pred.columns=quantiles
    
    return LGBM_models, LGBM_actual_pred

# Target1
models_1, results_1 = train_data(x_train_1, y_train_1, x_val_1, y_val_1,x_test)
results_1.sort_index()[:48]

# Target2
models_2, results_2 = train_data(x_train_2, y_train_2, x_val_2, y_val_2, x_test)
results_2.sort_index()[:48]

# print(results_1.shape, results_2.shape)
submission.iloc[:48]
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = results_1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = results_2.sort_index().values
# print(submission)
submission.iloc[:48]
submission.iloc[48:96]

submission.to_csv('./데이콘/태양열/data/submission_0120_mylgbm_1.csv', index=False)












import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import LSTM,Dense,Dropout,LayerNormalization,BatchNormalization,Input,concatenate,Reshape
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
# 딥러닝 BINARY
modelpath_2 = './데이콘/태양열/data/model_binary.h5'

def split_x(data,x_row,x_col):
    x = []
    for i in range(len(data)-x_row+1):
        subset = data[i:(i+x_row),0:x_col]
        x.append(subset)
    return np.array(x)

def split_y(data,y_row,y_col):
    y = []
    for i in range(len(data)-y_row+1):
        subset = data[(i):(i+y_row),y_col:-1]
        y.append(subset)
    return np.array(y)

drop_columns = ['Day','Hour','Minute']
df = pd.read_csv('./데이콘/태양열/data/train/train.csv')
sunup=[]
for i in range(len(df)):
    temp = df.iloc[i,-1]
    if temp>0.0:
        sunup.append(1)
    else :
        sunup.append(0)
df['Sun_up']=sunup

# Time 추가
Time=[]
for i in range(len(df)):
    temp = 60*df.iloc[i,1]+df.iloc[i,2]
    Time.append(temp)
df['Time']=Time
x_col,x_row,y_col,y_row = 9,336,-3,96
y = split_y(df.values,y_row,y_col)
y1=y[:,:,0].reshape(-1,96,)
y2=y[:,:,1].reshape(-1,96,)
print(df)
print(y.shape)
df.drop(drop_columns,axis='columns',inplace=True)
datasets = df.values[:,:]
scaler = MinMaxScaler()
datasets = scaler.fit_transform(datasets)
x = split_x(datasets,x_row,x_col)[:-96]
print(x.shape)
print(y.shape)
modelpath = './데이콘/태양열/data/일단킵.h5'
modelpath_2 = './데이콘/태양열/data/model_binary.h5'
model = load_model(modelpath)
model2 = load_model(modelpath_2)
result = []
binary = []
for i in range(81):
    # 데이터 로드
    filename = './데이콘/태양열/data/test/{}.csv'.format(i)
    df=pd.read_csv(filename)
    # sun_up 추가
    sun_up=[]
    Time=[]
    for i in range(len(df)):
        temp = 60*df.iloc[i,1]+df.iloc[i,2]
        Time.append(temp)
        temp = df.iloc[i,-1]
        if temp>0.0:
            sun_up.append(1)
        else :
            sun_up.append(0)
    df['Sun_up']=sun_up
    df['Time']=Time
    df.drop(drop_columns,axis='columns',inplace=True)
    datasets = df.values[:,:]
    datasets = scaler.transform(datasets)
    result.append(datasets)
    binary.append(datasets[:,-2].reshape(7,48))

a = np.array(result)
binary = np.array(binary)
predict = model.predict(a)
print(predict[0].shape)
predict2 = np.round(model2.predict(binary).reshape(-1,))

binary_multiply = pd.read_csv('./데이콘/태양열/data/submission_0120_mylgbm_1.csv')
dataframe = pd.read_csv('./데이콘/태양열/data/sample_submission.csv', encoding='cp949')


submission = pd.read_csv('./데이콘/태양열/data/submission_0120_mylgbm_1.csv', encoding='cp949',index_col=0)
ttt = pd.read_csv('./데이콘/태양열/data/sample_submission.csv', encoding='cp949',index_col=0)
print(submission)
print(ttt)
print(ttt.iloc[0,0])
for i in [0,1,2,3,4,5,6,7,8]:
    ttt.iloc[:,i]=submission.iloc[:,i]*predict2

submission.to_csv('./데이콘/태양열/data/LGBM_addBinary.csv', index=False)