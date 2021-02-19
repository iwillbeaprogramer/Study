batch_size = 256
patience=10
epochs = 10000
validation_split = 0.2

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import LSTM,Dense,Dropout,LayerNormalization,BatchNormalization,Input,concatenate,Reshape
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler,StandardScaler
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

x_col,x_row,y_col,y_row = 9,336,-4,96


# ['Day', 'Hour', 'Minute', 'DHI', 'DNI', 'WS', 'RH', 'T', 'TARGET','Sun_up]
drop_columns = ['Day','Hour','Minute']
df = pd.read_csv('./데이콘/태양열/data/train/train.csv')

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
print(df)
print(y.shape)
df.drop(drop_columns,axis='columns',inplace=True)
datasets = df.values[:,:]
scaler = StandardScaler()
datasets = scaler.fit_transform(datasets)

x = split_x(datasets,x_row,x_col)[:-96]
print(x.shape)
print(y.shape)



# modelpath = './데이콘/태양열/data/일단킵.h5'
modelpath_2 = './데이콘/태양열/data/model_binary.h5'
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
#model = load_model(modelpath)
model2 = load_model(modelpath_2)




result = []
binary = []

for i in range(81):
    # 데이터 로드
    filename = './데이콘/태양열/data/test/{}.csv'.format(i)
    df=pd.read_csv(filename)
    # sun_up 추가
    ghi=[]
    sun_up=[]
    Time=[]
    for i in range(len(df)):
        temp = np.cos(np.pi/2-np.abs(df.iloc[i,1]%12 - 6)/6*np.pi/2)
        ghi.append(temp)
        temp = 60*df.iloc[i,1]+df.iloc[i,2]
        Time.append(temp)
        temp = df.iloc[i,-1]
        if temp>0.0:
            sun_up.append(1)
        else :
            sun_up.append(0)
    df['cos'] = ghi
    df.insert(1,'GHI',df['DNI']*df['cos']+df['DHI'])
    df.drop(['cos'], axis= 1, inplace = True)
    df['Sun_up']=sun_up
    df['Time']=Time
    df.drop(drop_columns,axis='columns',inplace=True)
    datasets = df.values[:,:]
    datasets = scaler.transform(datasets)
    result.append(datasets.reshape(7,48,9))
    binary.append(datasets[:,-2].reshape(7,48))

a = np.array(result)
a=[]
for i in result:
    c=[]
    for j in range(len(i)):
        c.append(i[j])
    a.append(np.array(c))
a=np.array(a)
#binary = np.array(binary)
#predict = model.predict(a)
#print(predict[0].shape)
#predict2 = np.round(model2.predict(binary).reshape(-1,1))


#a = predict[0].reshape(-1,1)
#b = np.round(predict2)
#print(a.shape)

submit = pd.read_csv('./데이콘/태양열/data/sample_submission.csv',index_col=0)
for i in [1,2,3,4,5,6,7,8,9]:
    path = './데이콘/태양열/data/01_22/{}_concat_all_home.h5'.format(i)
    # path = './데이콘/태양열/data/01_21//{}Standard_LNormalizationO_model0120concat_batch48_epoch60_validation_split0.2.h5'.format(i)
    model = load_model(path,compile=False)
    k = model.predict(a)
    submit.iloc[:,i-1] = k.reshape(-1,1) # *predict2

submit.to_csv('./데이콘/태양열/data/01_22/submit_file_0121.csv')


































'''
index = []
for i in range(len(a)):
    number = (i)//48
    day = (i//48)%2+7
    hour = (i//2)-((i//2)//24)*24
    if i%2==0:
        minute = str(0)+str(0)
    else:
        minute = str(30)
    indexname = '{}.csv_Day{}_{}h{}m'.format(str(number),str(day),str(hour),minute)
    index.append(indexname)

count=0
for i in index:
    count+=1
    print(i)
    if(count>=120):
        break;
print(len(index))
'''


