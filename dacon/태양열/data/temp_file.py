batch_size = 256
patience=10
epochs = 60
validation_split = 0.2
lr = 0.015

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import LSTM,Dense,Dropout,LayerNormalization,BatchNormalization,Input,concatenate,Reshape,Conv1D,GRU
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tensorflow.keras.backend import mean, maximum
from tensorflow.keras.optimizers import Adam,Adadelta,Adamax,Adagrad
optimizer = Adam(lr=lr)

def quantile_loss(q, y, pred):
    err = (y-pred)
    return mean(maximum(q*err, (q-1)*err), axis=-1)

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

x_col,x_row,y_col,y_row = 9,336,-3,96


# ['Day', 'Hour', 'Minute', 'DHI', 'DNI', 'WS', 'RH', 'T', 'TARGET','Sun_up]
drop_columns = ['Day','Hour','Minute']
df = pd.read_csv('./데이콘/태양열/data/train/train.csv')

# sunup 추가
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

y = split_y(df.values,y_row,y_col)
y1=y[:,:,0].reshape(-1,96,)
y2=y[:,:,1].reshape(-1,96,)
print(df)
print(y.shape)
df.drop(drop_columns,axis='columns',inplace=True)
datasets = df.values[:,:]
scaler = StandardScaler()
datasets = scaler.fit_transform(datasets)

x = split_x(datasets,x_row,x_col)[:-96]
print(x.shape)
print(y.shape)


q_lst = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for q in q_lst:
    modelpath = './데이콘/태양열/data/01_20/{}GRU_O_model0120concat_batch{}_epoch{}_validation_split{}_________.h5'.format(3,batch_size,epochs,validation_split)
    cp = ModelCheckpoint(monitor='val_loss',filepath = modelpath,save_best_only=True)
    es = EarlyStopping(monitor='val_loss',patience=patience)
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',patience=patience/2,factor=0.85)
    inputs = Input(shape=(336,8))
    conv1d = Conv1D(256,kernel_size=2,padding='valid')(inputs)
    conv1d = Conv1D(128,kernel_size=2,padding='valid')(conv1d)
    conv1d = Conv1D(64,kernel_size=2,padding='valid')(conv1d)
    lstm = GRU(32,activation='relu')(conv1d)
    dense = Dense(512,activation='relu')(lstm)
    dense = Dense(256,activation='relu')(dense)
    outputs1 = Dense(96)(dense)
    model = Model(inputs = inputs , outputs=outputs1)
    model.compile(loss = lambda y,pred: quantile_loss(q,y,pred), optimizer=optimizer, metrics=['mse'])
    model.fit(x,y1,validation_split=validation_split,epochs=epochs,batch_size=batch_size,verbose=1,callbacks=[es,cp,reduce_lr])