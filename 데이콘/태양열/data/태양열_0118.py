import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout,LayerNormalization,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
def split_x(data,x_row,x_col):
    x = []
    for i in range(len(data)-x_row+1):
        subset = data[i:(i+x_row),1:x_col]
        x.append(subset)
    return np.array(x)

def split_y(data,y_row,y_col):
    y = []
    for i in range(len(data)-y_row+1):
        subset = data[(i):(i+y_row),y_col]
        y.append(subset)
    return np.array(y)

x_col,x_row,y_col,y_row = 8,336,-1,96


df = pd.read_csv('./데이콘/태양열/data/train/train.csv')
datasets = df.values[:,:-1]
scaler = MinMaxScaler()
datasets = scaler.fit_transform(datasets)

x = split_x(datasets,x_row,x_col)[:-96]
y = split_y(df.values,y_row,y_col)[336:]


batch_size = 4
patience=10
epochs = 10000
validation_split = 0.2
modelpath = './데이콘/태양열/data/model_batch{}_epoch{}_validation_split{}.h5'.format(batch_size,epochs,validation_split)
cp = ModelCheckpoint(monitor='val_loss',filepath = modelpath,save_best_only=True)
es = EarlyStopping(monitor='val_loss',patience=patience)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',patience=patience/2,factor=0.5)

model = Sequential()
model.add(LSTM(32,activation='relu',input_shape=(336,7)))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='relu'))
model.compile(loss = 'mse',optimizer='adam',metrics=['mse'])
model.fit(x,y,validation_split=validation_split,epochs=epochs,batch_size=batch_size,verbose=1,callbacks=[es,cp,reduce_lr])




