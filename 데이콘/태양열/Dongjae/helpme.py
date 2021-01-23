import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

#### 매일 같은 시간대의 데이터끼리만 묶어서 시계열로 만드는 모델! 훈련시간 거의 두시간;;;

train = pd.read_csv('./데이콘/태양열/data/train/train.csv')
submission = pd.read_csv('./데이콘/태양열/data/sample_submission.csv')

day = 6 # 시계열로 만들 일수!! 여기서 조정해준다!!

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

 # 48스텝씩 건너뛰면서 같은 시간대끼리만 묶어주는 함수!
 # 예를들어 1095 일치의 트레인 데이터를 넣으면
 # (00시00분*1095,칼럼수) ~ (23시30분*1095,칼럼수) 이런 묶음으로 잘라준다!!
 # 주의: 데이터프레임을 넣으면 np.array 로 나온다!!
def split_to_seq(data): 
    tmp = []
    for i in range(48):
        tmp1 = pd.DataFrame()
        for j in range(int(len(data)/48)):
            tmp2 = data.iloc[j*48+i,:]
            tmp2 = tmp2.to_numpy()
            tmp2 = tmp2.reshape(1,tmp2.shape[0])
            tmp2 = pd.DataFrame(tmp2)
            tmp1 = pd.concat([tmp1,tmp2])
        x = tmp1.to_numpy()
        tmp.append(x)
    return np.array(tmp)

# 특정 열이 해가 뜨고 해가지는 시간을 가지고 각 시간의 cos를 계산해주는 함수!!
# 해가 뜨는시간에 90도, 지는시간에 90도, 중간에는 0도!!
def make_cos(dataframe): 
    dataframe /=dataframe
    c = dataframe.dropna()
    d = c.to_numpy()

    def into_cosine(seq):
        for i in range(len(seq)):
            if i < len(seq)/2:
                seq[i] = float((len(seq)-1)/2) - (i)
            if i >= len(seq)/2:
                seq[i] = seq[len(seq) - i - 1]
        seq = seq/ np.max(seq) * np.pi/2
        seq = np.cos(seq)
        return seq

    d = into_cosine(d)
    dataframe = dataframe.replace(to_replace = np.NaN, value = 0)
    dataframe.loc[dataframe['cos'] == 1] = d
    return dataframe

# 베이스라인을 베껴왔다
# 1. 코스를 넣어서 GHI계산하고
# 2. 트레인이면 타겟값들을 붙여주고 테스트면 위에서 정한 일수*48 데이터를 리턴한다!!
def preprocess_data(data, is_train = True):
    a = pd.DataFrame()
    for i in range(int(len(data)/48)):
        tmp = pd.DataFrame()
        tmp['cos'] = data.loc[i*48:(i+1)*48-1,'TARGET']
        tmp['cos'] = make_cos(tmp)
        a = pd.concat([a,tmp])
    data['cos'] = a
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    temp = data.copy()
    temp = temp[['TARGET','GHI','DHI','DNI','RH','T']]

    if is_train == True:
        temp['TARGET1'] = temp['TARGET'].shift(-48).fillna(method = 'ffill')
        temp['TARGET2'] = temp['TARGET'].shift(-96).fillna(method = 'ffill')
        temp = temp.dropna()
        return temp.iloc[:-96]

    elif is_train == False:
        return temp.iloc[-48*day:, :]

df_train = preprocess_data(train)
scale.fit(df_train.iloc[:,:-2])
df_train.iloc[:,:-2] = scale.transform(df_train.iloc[:,:-2])

# 81개의 테스트일수를 (81, 48, 일수, 8) 로 나눠준다
# 추후에 48개의 모델에 81번 돌린다!! 인풋 : (1, 일수, 6)
df_test = []
for i in range(81):
    file_path = './데이콘/태양열/data/test/%d.csv'%i
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp,is_train=False)
    temp = scale.transform(temp)
    temp = pd.DataFrame(temp)
    temp = split_to_seq(temp)
    df_test.append(temp)

test = np.array(df_test)
train = split_to_seq(df_train)
# print(train.shape) #(48, 1093, 10)
# print(test.shape) #(81, 48, 4, 6)

def split_xy(data,timestep):
    x, y1, y2 = [],[],[]
    for i in range(len(data)):
        x_end = i + timestep
        if x_end>len(data):
            break
        tmp_x = data[i:x_end,:-2]
        tmp_y1 = data[x_end-1:x_end,-2]
        tmp_y2 = data[x_end-1:x_end,-1]
        x.append(tmp_x)
        y1.append(tmp_y1)
        y2.append(tmp_y2)
    return(np.array(x),np.array(y1),np.array(y2))

# 모델을 48번 돌릴것이기 때문에 트레인x 를 (48, 훈련수, 일수, 6) 로 잘라줄 것이다~!
x,y1,y2 = [],[],[]
for i in range(48):
    tmp1,tmp2,tmp3 = split_xy(train[i],day)
    x.append(tmp1)
    y1.append(tmp2)
    y2.append(tmp3)

# 이러면 x, y1, y2 에 각각 >> 결국 인풋은 (일수,8) 아웃풋은 (1,) 쉐이프!!
x = np.array(x) # (48, 훈련수, 일수, 6)
y1 = np.array(y1) # (48, 훈련수, 1)
y2 = np.array(y2) # (48, 훈련수, 1)

from sklearn.model_selection import train_test_split as tts

def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)
quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

#2. 모델링
from tensorflow.keras.models import load_model

for i in range(48):
    print(f'{int(i/2)}시 {i%2*30}분 시간대 진행중...')
    # x_train, x_val, y1_train, y1_val, y2_train, y2_val = tts(x[i],y1[i],y2[i], train_size = 0.7,shuffle = True, random_state = 0)
    # 내일!
    for j in quantiles:
        filepath_cp = f'./데이콘/태양열/Dongjae/_dacon_{i:2d}_y1seq_{j:.1f}.hdf5'
        model = load_model(filepath_cp, compile = False)
        model.compile(loss = lambda y_true,y_pred: quantile_loss(j,y_true,y_pred), optimizer = 'adam', metrics = [lambda y,y_pred: quantile_loss(j,y,y_pred)])
        x = []
        for k in range(81):
            x.append(test[k,i])
        x = np.array(x)
        df_temp1 = pd.DataFrame(model.predict(x).round(2))
        # df_temp1 = pd.concat(pred, axis = 0)
        df_temp1[df_temp1<0] = 0
        num_temp1 = df_temp1.to_numpy()
        if i%2 == 0:
            submission.loc[submission.id.str.contains(f"Day7_{int(i/2)}h00m"), [f"q_{j:.1f}"]] = num_temp1
        elif i%2 == 1:
            submission.loc[submission.id.str.contains(f"Day7_{int(i/2)}h30m"), [f"q_{j:.1f}"]] = num_temp1

    # 모레!
    for j in quantiles:
        filepath_cp = f'./데이콘/태양열/__Dongjae/dacon_{i:2d}_y1seq_{j:.1f}.hdf5'
        model = load_model(filepath_cp, compile = False)
        model.compile(loss = lambda y_true,y_pred: quantile_loss(j,y_true,y_pred), optimizer = 'adam', metrics = [lambda y,y_pred: quantile_loss(j,y,y_pred)])
        x = []
        for k in range(81):
            x.append(test[k,i])
        x = np.array(x)
        df_temp2 = pd.DataFrame(model.predict(x).round(2))
        # df_temp1 = pd.concat(pred, axis = 0)
        df_temp2[df_temp2<0] = 0
        num_temp2 = df_temp2.to_numpy()
        if i%2 == 0:
            submission.loc[submission.id.str.contains(f"Day8_{int(i/2)}h00m"), [f"q_{j:.1f}"]] = num_temp2
        elif i%2 == 1:
            submission.loc[submission.id.str.contains(f"Day8_{int(i/2)}h30m"), [f"q_{j:.1f}"]] = num_temp2

submission.to_csv('./데이콘/태양열/Dongjae/0122_timeseries_scale.csv', index = False)