batch_size = 48
patience=4
epochs = 40
validation_split = 0.15
lr = 0.025
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import LSTM,Dense,Dropout,LayerNormalization,BatchNormalization,Input,concatenate,Reshape,Conv1D,BatchNormalization,Activation,MaxPooling1D,Flatten,Conv2D,MaxPooling2D
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

x_col,x_row,y_col,y_row = 9,336,-4,96


# ['Day', 'Hour', 'Minute', 'DHI', 'DNI', 'WS', 'RH', 'T', 'TARGET','Sun_up]
drop_columns = ['Day','Hour','Minute']
df = pd.read_csv('./데이콘/태양열/data/train/train.csv')

# GHI
ghi = []
for i in range(len(df)):
    temp = np.cos(np.pi/2-np.abs(df.iloc[i,1]%12 - 6)/6*np.pi/2)
    ghi.append(temp)
df['cos'] = ghi
df.insert(1,'GHI',df['DNI']*df['cos']+df['DHI'])
df.drop(['cos'], axis= 1, inplace = True)


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
df.drop(drop_columns,axis='columns',inplace=True)
datasets = df.values[:,:]
scaler = StandardScaler()
datasets = scaler.fit_transform(datasets)
x = split_x(datasets,x_row,x_col)[:-96]
w=[]
for i in range(len(x)):
    e=[]
    for j in range(0,335,48):
        e.append(x[i][j:(j+48),:])
    w.append(np.array(e))
w=np.array(w)
print(w.shape)
q_lst = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i,q in enumerate(q_lst):
    modelpath = './데이콘/태양열/data/01_22/{}_concat_all_home.h5'.format(str(int(q*10)),batch_size,epochs,validation_split)
    cp = ModelCheckpoint(monitor='val_loss',filepath = modelpath,save_best_only=True)
    es = EarlyStopping(monitor='val_loss',patience=patience)
    #reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',patience=patience/2,factor=0.85)
    model = Sequential()
    model.add(Conv2D(64,2,padding='same',input_shape = (7,48,9),activation='relu',data_format='channels_last'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(32,2,padding='same',activation='relu'))
    model.add(Flatten())
    model.add(Dense(96,activation='relu'))
    model.add(Dense(96,activation='relu'))
    
    model.compile(loss = lambda y,pred: quantile_loss(q,y,pred), optimizer=optimizer, metrics=['mse'])
    model.fit(w,y1,validation_split=validation_split,epochs=epochs,batch_size=batch_size,verbose=1,callbacks=[es,cp])


#1 : 1.5
#2 : 2.6   단순모델 윈
#3 : 3.1
#4 : 7.3
#5 : 9.2
#6 : 3.1
#7 : 2.5
#8 : 2.5
#9 : 1.4

#549



'''
1 : 1.5
2 : 3.6
3 : 5.5
4 : 3.2
5 : 3.0
6 : 2.5
7 : 2.2
8 : 1.6
9 : 0.8
'''



'''
1 - 1.5
2 - 3.1
3 - 4.7
4 - 3.0
5 - 7.9
6 - 9.3
7 - 2.0
8 - 1.5
9 - 1.1
'''