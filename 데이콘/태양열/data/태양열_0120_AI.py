batch_size = 256
patience=10
epochs = 10000
validation_split = 0.2

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import LSTM,Dense,Dropout,LayerNormalization,BatchNormalization,Input,concatenate,Reshape
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
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



modelpath = './데이콘/태양열/data/model0120concat_batch{}_epoch{}_validation_split{}.h5'.format(batch_size,epochs,validation_split)
cp = ModelCheckpoint(monitor='val_loss',filepath = modelpath,save_best_only=True)
es = EarlyStopping(monitor='val_loss',patience=patience)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',patience=patience/2,factor=0.5)
inputs = Input(shape=(336,8))
lstm = LSTM(16,activation='relu')(inputs)
dense = Dense(512,activation='relu')(lstm)
dense = Dense(256,activation='relu')(dense)


dense1 = Dense(128,activation='relu')(dense)
outputs1 = Dense(96,name='output1')(dense1)

dense2 = Dense(128,activation='sigmoid')(dense)
outputs2 = Dense(96,activation='sigmoid',name='output2')(dense2)
model = Model(inputs = inputs , outputs=[outputs1,outputs2])
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

model.compile(loss = ['mse','binary_crossentropy'],optimizer='adam',metrics=['mse','accuracy'])
model.fit(x,[y1,y2],validation_split=validation_split,epochs=epochs,batch_size=batch_size,verbose=1,callbacks=[es,cp,reduce_lr])



