batch_size = 256
patience=10
epochs = 10000
validation_split = 0.2

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import LSTM,Dense,Dropout,LayerNormalization,BatchNormalization,Input,concatenate,Reshape
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
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
scaler = MinMaxScaler()
datasets = scaler.fit_transform(datasets)

x = split_x(datasets,x_row,x_col)[:-96]
print(x.shape)
print(y.shape)



modelpath = './데이콘/태양열/data/일단킵.h5'
'''
lstm = LSTM(16,activation='relu')(inputs)
dense = Dense(512,activation='relu')(lstm)
dense = Dense(256,activation='relu')(dense)


dense1 = Dense(128,activation='relu')(dense)
outputs1 = Dense(96,name='output1')(dense1)

dense2 = Dense(128,activation='sigmoid')(dense)
outputs2 = Dense(96,activation='sigmoid',name='output2')(dense2)
model = Model(inputs = inputs , outputs=[outputs1,outputs2])
'''
'''
model = Sequential()
model.add(LSTM(32,activation='relu',input_shape=(336,8)))
model.add(Dense(1024,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
'''

#model.compile(loss = ['mse','binary_crossentropy'],optimizer='adam',metrics=['mse','accuracy'])
#model.fit(x,[y1,y2],validation_split=validation_split,epochs=epochs,batch_size=batch_size,verbose=1,callbacks=[es,cp,reduce_lr])
model = load_model(modelpath)





result = []
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

a = np.array(result)
predict = model.predict(a)


a = predict[0].reshape(-1,1)

import matplotlib.pyplot as plt
plt.plot(a[480:720])
plt.grid()
plt.show()


