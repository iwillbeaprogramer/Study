import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,GRU
import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

seed=300
#seed=221
epochs=200
batch_size=4

N = 10000 # number of timesteps
T = 500 # time will vary from 0 to T with step delt
ts = np.linspace(0,T,N+1)
delt = T/N
gamma = .05 # damping, 0 is no damping

A = np.zeros((2,2))
B = np.zeros((2,1))
C = np.zeros((1,2))

A[0,0] = 1
A[0,1] = (1-gamma*delt/2)*delt
A[1,1] = 1 - gamma*delt

B[0,0] = delt**2/269
B[1,0] = delt

C[0,0] = 1

np.random.seed(seed)

x = np.zeros((2,N+1))
x[:,0] = [0,0]
y = np.zeros((1,N))

w = np.random.randn(1,N)
n = np.random.randn(1,N)

for t in range(N):
    y[:,t] = C.dot(x[:,t]) + n[:,t]
    x[:,t+1] = A.dot(x[:,t]) + B.dot(w[:,t])
    
    
x_true = x.copy()
w_true = w.copy()
n_true = n.copy()

plt.figure(figsize=(8,6), dpi=100)
plt.subplot(2,1,1)
plt.plot(ts,x[0,:], label='position (true)')
plt.plot(ts[:-1],y[0,:], alpha=0.5, label='position (measured)')
plt.ylabel(r'$p$')
plt.legend()
plt.grid()
plt.subplot(2,1,2)
plt.plot(ts,x[1,:], label='velocity (true)')
plt.xlabel('time')
plt.ylabel(r'$v$')
plt.legend()
plt.grid()
#plt.show()

plt.figure(figsize=(8,6), dpi=100)
plt.subplot(2,1,1)
plt.plot(ts[:-1],w_true[0,:], label='wind force (true)')
plt.ylabel(r"$w$")
plt.legend()
plt.grid()
plt.subplot(2,1,2)
plt.plot(ts[:-1],n_true[0,:], label='sensor noise (true)')
plt.xlabel('time')
plt.ylabel(r'$n$')
plt.legend()
plt.grid()
#plt.show()

sigma_w = B@B.T
sigma_v = 1
hat_x = np.zeros((2,N))
hat_x[:,0] = np.array([0,0])
sigma = np.eye(2)*10

x_train = np.zeros((2,N))
y_train = np.zeros((2,N))

for t in range(N-1):
    K = A@sigma@C.T@np.linalg.inv(C@sigma@C.T+sigma_v)
    hat_x[:,t+1] = A@hat_x[:,t] + K@(y[0,t] - C@hat_x[:,t])
    sigma = A@sigma@A.T - K@C@sigma@A.T+sigma_w
    x_train[:,t] = (y[0,t] - C@hat_x[:,t])
    y_train[:,t] = hat_x[:,t+1]-A@hat_x[:,t]
x_train = x_train.reshape(-1,2)
y_train = y_train.reshape(-1,2)


plt.figure(figsize = (8,6),dpi=100)
plt.subplot(2,1,1)
plt.plot(ts,x[0,:],label='position (true)')
plt.plot(ts[:-1],hat_x[0,:],alpha = 0.5, label='position (Kalman estimate)')
plt.ylabel(r'$p$')
plt.title('Kalman estimates')
plt.legend()
plt.grid()
plt.subplot(2,1,2)
plt.plot(ts,x[1,:],label='velocity (true)')
plt.plot(ts[:-1],hat_x[1,:],alpha=1,label='velocity (Kalman estimates)')
plt.grid()

Kalman_RMSE_list_G=[]
DeepLearning_RMSE_list_G_1=[]
DeepLearning_RMSE_list_G_2=[]
n_list=[]

N = 10000 # number of timesteps
T = 500 # time will vary from 0 to T with step delt
ts = np.linspace(0,T,N+1)
delt = T/N
gamma = .05 # damping, 0 is no damping

A = np.zeros((2,2))
B = np.zeros((2,1))
C = np.zeros((1,2))

A[0,0] = 1
A[0,1] = (1-gamma*delt/2)*delt
A[1,1] = 1 - gamma*delt

B[0,0] = delt**2/269
B[1,0] = delt

C[0,0] = 1

np.random.seed(seed)

x = np.zeros((2,N+1))
x[:,0] = [0,0]
y = np.zeros((1,N))

w = np.random.randn(1,N)
n = np.random.randn(1,N)


for t in range(N):
    switch = np.random.randint(0,10)
    switch_ = np.random.randint(0,10)
    if switch==7 or switch==3 or switch==5 or switch==0:
        var = np.random.randint(0,5)
    else:
        var=0
        
    if switch_==3 or switch_==0 or switch_==9 or switch_==7:
        var_ = np.random.randint(0,2)
    else:
        var_=0
    
    w[:,t] = var_*w[:,t]
    n[:,t] = var*n[:,t]
    y[:,t] = C.dot(x[:,t]) + n[:,t]
    x[:,t+1] = A.dot(x[:,t]) + B.dot(w[:,t])
    
    
x_true = x.copy()
w_true = w.copy()
n_true = n.copy()

plt.figure(figsize=(8,6), dpi=100)
plt.subplot(2,1,1)
plt.plot(ts,x[0,:], label='position (true)')
plt.plot(ts[:-1],y[0,:], alpha=0.5, label='position (measured)')
plt.ylabel(r'$p$')
plt.legend()
plt.grid()
plt.subplot(2,1,2)
plt.plot(ts,x[1,:], label='velocity (true)')
plt.xlabel('time')
plt.ylabel(r'$v$')
plt.legend()
plt.grid()
#plt.show()

plt.figure(figsize=(8,6), dpi=100)
plt.subplot(2,1,1)
plt.plot(ts[:-1],w_true[0,:], label='wind force (true)')
plt.ylabel(r"$w$")
plt.legend()
plt.grid()
plt.subplot(2,1,2)
plt.plot(ts[:-1],n_true[0,:], label='sensor noise (true)')
plt.xlabel('time')
plt.ylabel(r'$n$')
plt.legend()
plt.grid()
#plt.show()

N_test = 8000 # number of timesteps
T_test = 400 # time will vary from 0 to T with step delt
ts_test = np.linspace(0,T_test,N_test+1)
delt_test = T_test/N_test
gamma_test = .05 # damping, 0 is no damping

A_test = np.zeros((2,2))
B_test= np.zeros((2,1))
C_test = np.zeros((1,2))

A_test[0,0] = 1
A_test[0,1] = (1-gamma_test*delt_test/2)*delt_test
A_test[1,1] = 1 - gamma_test*delt_test

B_test[0,0] = delt_test**2/269
B_test[1,0] = delt_test

C_test[0,0] = 1


x_test = np.zeros((2,N_test+1))
x_test[:,0] = [0,0]
y_test = np.zeros((1,N_test))

w_test = np.random.randn(1,N_test)
n_test = np.random.randn(1,N_test)

for t in range(N_test):
    switch = np.random.randint(0,10)
    if switch==7 or switch==3 or switch==5:
        var = np.random.randint(0,10)
    else:
        var=0
    y_test[:,t] = C_test.dot(x_test[:,t]) + var*n_test[:,t]
    x_test[:,t+1] = A_test.dot(x_test[:,t]) + B_test.dot(w_test[:,t])
    
plt.figure(figsize=(8,6), dpi=100)
plt.subplot(2,1,1)
plt.plot(ts_test,x_test[0,:], label='position (true)')
plt.plot(ts_test[:-1],y_test[0,:], alpha=0.5, label='position (measured)')
plt.ylabel(r'$p$')
plt.legend()
plt.grid()
plt.subplot(2,1,2)
plt.plot(ts_test,x_test[1,:], label='velocity (true)')
plt.xlabel('time')
plt.ylabel(r'$v$')
plt.legend()
plt.grid()
#plt.show()


sigma_w = B@B.T
sigma_v = 1
hat_x = np.zeros((2,N))
hat_x[:,0] = np.array([0,0])
sigma = np.eye(2)*10

x_train = np.zeros((2,N))
y_train = np.zeros((2,N))

for t in range(N-1):
    K = A@sigma@C.T@np.linalg.inv(C@sigma@C.T+sigma_v)
    hat_x[:,t+1] = A@hat_x[:,t] + K@(y[0,t] - C@hat_x[:,t])
    sigma = A@sigma@A.T - K@C@sigma@A.T+sigma_w
    x_train[:,t] = (y[0,t] - C@hat_x[:,t])
    y_train[:,t] = hat_x[:,t+1]-A@hat_x[:,t]
x_train = x_train.reshape(-1,2)
y_train = y_train.reshape(-1,2)


plt.figure(figsize = (8,6),dpi=100)
plt.subplot(2,1,1)
plt.plot(ts,x[0,:],label='position (true)')
plt.plot(ts[:-1],hat_x[0,:],alpha = 0.5, label='position (Kalman estimate)')
plt.ylabel(r'$p$')
plt.title('Kalman estimates')
plt.legend()
plt.grid()
plt.subplot(2,1,2)
plt.plot(ts,x[1,:],label='velocity (true)')
plt.plot(ts[:-1],hat_x[1,:],alpha=1,label='velocity (Kalman estimates)')
plt.grid()

n_list=[]
end=201

'''
모델훈련
'''


'''
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
for n in range(2,201):
    print(n)
    modelpath_1 = './my_model/'+'/LSTM_N='+str(n)+'.hdf5'
    modelpath_2 = './my_model/'+'/LSTM_N='+str(n)+'_multilayer.hdf5'
    modelpath_3 = './my_model/'+'/GRU_N='+str(n)+'_multilayer.hdf5'
    checkpointer_1 = ModelCheckpoint(filepath=modelpath_1, monitor = 'loss',save_best_only=True)
    checkpointer_2 = ModelCheckpoint(filepath=modelpath_2, monitor = 'loss',save_best_only=True)
    checkpointer_3 = ModelCheckpoint(filepath=modelpath_3, monitor = 'loss',save_best_only=True)
    es1 = EarlyStopping(monitor='loss',patience=20,mode='auto')
    es2 = EarlyStopping(monitor='loss',patience=20,mode='auto')
    es3 = EarlyStopping(monitor='loss',patience=20,mode='auto')
    n_list.append(n)
    temp = []
    for i in range(len(y[0])-n):
        temp.append(y[0,i:i+n].reshape(n,-1))
    x_train = np.array(temp).reshape(-1,n,1)
    y_train = x[0,n:-1]
    model_1 = Sequential()
    model_1.add(LSTM(50,input_shape = (n,1)))
    model_1.add(Dense(1))
    model_1.compile(loss = 'mse',optimizer = 'adam',metrics=['mse'])
    history_1 = model_1.fit(x_train,y_train,epochs = 1,verbose = 1,batch_size=batch_size,callbacks=[checkpointer_1,es1])
    model_2 = Sequential()
    model_2.add(LSTM(50,input_shape = (n,1),activation='relu'))
    model_2.add(Dense(50,input_dim = n,activation='relu'))
    model_2.add(Dense(25,activation='relu'))
    model_2.add(Dense(5,activation='relu'))
    model_2.add(Dense(1))
    model_2.compile(loss = 'mse',optimizer = 'adam',metrics = ['mse'])
    history_2 = model_2.fit(x_train,y_train,epochs = epochs,verbose = 1,batch_size=batch_size,callbacks=[checkpointer_2,es2])
    model_3 = Sequential()
    model_3.add(GRU(50,input_shape = (n,1),activation='relu'))
    model_3.add(Dense(50,input_dim = n,activation='relu'))
    model_3.add(Dense(25,activation='relu'))
    model_3.add(Dense(5,activation='relu'))
    model_3.add(Dense(1))
    model_3.compile(loss = 'mse',optimizer = 'adam',metrics = ['mse'])
    history_3 = model_3.fit(x_train,y_train,epochs = epochs,verbose = 1,batch_size=batch_size,callbacks=[checkpointer_3,es3])
'''



'''
    딥러닝 생성
'''

Kalman_RMSE_list_N=[]
DeepLearning_RMSE_list_N_1=[]
DeepLearning_RMSE_list_N_2=[]
LSTM_only_RMSE_list=[]
LSTM_multi_RMSE_list=[]
GRU_multi_RMSE_list=[]
n_list=[]



##########            칼만 생성            ##########
for n in range(2,13):
    n_list.append(n)
    sigma_w = B_test@B_test.T
    sigma_v = 1
    hat_x = np.zeros((2,N_test))
    hat_x[:,0] = np.array([0,0])
    sigma = np.eye(2)*10
    for t in range(N_test-1):
        K = A_test@sigma@C_test.T@np.linalg.inv(C_test@sigma@C_test.T+sigma_v)
        hat_x[:,t+1] = A_test@hat_x[:,t] + K@(y_test[0,t] - C_test@hat_x[:,t])
        sigma = A_test@sigma@A_test.T - K@C_test@sigma@A_test.T+sigma_w


    ##########            딥러닝 생성1            ##########
    test = []
    for i in range(len(y_test[0])-n):
        test.append(y_test[0,i:i+n].reshape(n,-1))
    x_test_ready_2 = np.array(test).reshape(-1,n)
    y_test_ready_2 = x_test[0,n:-1]
    x_test_ready_2_rnn = np.array(test).reshape(-1,n,1)
    #model_1=tf.keras.models.load_model('./my_model/N='+str(n)+'.hdf5')
    #model_2=tf.keras.models.load_model('./my_model/N='+str(n)+'(multilayer).hdf5')
    model_3=tf.keras.models.load_model('./my_model/LSTM_N='+str(n)+'.hdf5')
    model_4=tf.keras.models.load_model('./my_model/LSTM_N='+str(n)+'_multilayer.hdf5')
    model_5=tf.keras.models.load_model('./my_model/GRU_N='+str(n)+'_multilayer.hdf5')
    #position_hat_1 = model_1.predict(x_test_ready_2)
    #position_hat_2 = model_2.predict(x_test_ready_2)
    ##########            RMSE                  ###########
    position_hat_3 = model_3.predict(x_test_ready_2_rnn)
    position_hat_4 = model_4.predict(x_test_ready_2_rnn)
    position_hat_5 = model_5.predict(x_test_ready_2_rnn)

    
    ##########            RMSE                  ###########
    Kalman_RMSE = 0
    DeepLearning_RMSE_1 = 0
    DeepLearning_RMSE_2 = 0
    LSTM_only_RMSE=0
    LSTM_multi_RMSE=0
    GRU_multi_RMSE=0
    for k in range(len(y_test[0])-n):
        Kalman_RMSE += (x_test[0,n:-1][k]-hat_x[0,n:][k])**2
        #DeepLearning_RMSE_1 += (x_test[0,n:-1][k]-position_hat_1[k])**2
        #DeepLearning_RMSE_2 += (x_test[0,n:-1][k]-position_hat_2[k])**2
        LSTM_only_RMSE += (x_test[0,n:-1][k]-position_hat_3[k])**2
        LSTM_multi_RMSE += (x_test[0,n:-1][k]-position_hat_4[k])**2
        GRU_multi_RMSE += (x_test[0,n:-1][k]-position_hat_5[k])**2
        
    Kalman_RMSE = (Kalman_RMSE/(len(y_test[0])-n))**0.5
    #DeepLearning_RMSE_1 = (DeepLearning_RMSE_1/(len(y_test[0])-n))**0.5
    #DeepLearning_RMSE_2 = (DeepLearning_RMSE_2/(len(y_test[0])-n))**0.5
    LSTM_only_RMSE = (LSTM_only_RMSE/(len(y_test[0])-n))**0.5
    LSTM_multi_RMSE = (LSTM_multi_RMSE/(len(y_test[0])-n))**0.5
    GRU_multi_RMSE = (GRU_multi_RMSE/(len(y_test[0])-n))**0.5
    Kalman_RMSE_list_N.append(Kalman_RMSE)
    #DeepLearning_RMSE_list_N_1.append(DeepLearning_RMSE_1[0])
    #DeepLearning_RMSE_list_N_2.append(DeepLearning_RMSE_2[0])
    LSTM_only_RMSE_list.append(LSTM_only_RMSE)
    LSTM_multi_RMSE_list.append(LSTM_multi_RMSE)
    GRU_multi_RMSE_list.append(GRU_multi_RMSE)

more = []
for c in range(len(Kalman_RMSE_list_N)):
    if Kalman_RMSE_list_N[c]==min(Kalman_RMSE_list_N[c],LSTM_only_RMSE_list[c],LSTM_multi_RMSE_list[c],GRU_multi_RMSE_list[c]):
        more.append('KalmanFilter Good')
#    elif DeepLearning_RMSE_list_N_1[c]==min(Kalman_RMSE_list_N[c],DeepLearning_RMSE_list_N_1[c],DeepLearning_RMSE_list_N_2[c],LSTM_only_RMSE_list[c],LSTM_multi_RMSE_list[c],GRU_multi_RMSE_list[c]):
#        more.append('DeepLearning_1 Good')
    elif LSTM_only_RMSE_list[c]==min(Kalman_RMSE_list_N[c],LSTM_only_RMSE_list[c],LSTM_multi_RMSE_list[c],GRU_multi_RMSE_list[c]):
        more.append('LSTM_only Good')
    elif LSTM_multi_RMSE_list[c]==min(Kalman_RMSE_list_N[c],LSTM_only_RMSE_list[c],LSTM_multi_RMSE_list[c],GRU_multi_RMSE_list[c]):
        more.append('LSTM_multi Good')
    elif GRU_multi_RMSE_list[c]==min(Kalman_RMSE_list_N[c],LSTM_only_RMSE_list[c],LSTM_multi_RMSE_list[c],GRU_multi_RMSE_list[c]):
        more.append('GRU_multi Good')
#    else :
#        more.append('DeepLearning_2 Good')
result_dic = {
    'N': n_list,
    'Kalman RMSE':Kalman_RMSE_list_N,
    #'DeepLearning RMSE_1':DeepLearning_RMSE_list_N_1,
    #'DeepLearning RMSE_2':DeepLearning_RMSE_list_N_2,
    'LSTM_only':LSTM_only_RMSE_list,
    'LSTM_multi':LSTM_multi_RMSE_list,
    'GRU_multi':GRU_multi_RMSE_list,
    'Better':more
}
df_N = pd.DataFrame(result_dic)
print("Best case DeepLearning")
#print(df_N.iloc[DeepLearning_RMSE_list_N_1.index(min(DeepLearning_RMSE_list_N_1))])
#print(df_N.iloc[DeepLearning_RMSE_list_N_2.index(min(DeepLearning_RMSE_list_N_2))])
print(df_N.iloc[LSTM_only_RMSE_list.index(min(LSTM_only_RMSE_list))])
print(df_N.iloc[LSTM_multi_RMSE_list.index(min(LSTM_multi_RMSE_list))])
print(df_N.iloc[GRU_multi_RMSE_list.index(min(GRU_multi_RMSE_list))])
df_N.to_csv('Dense.csv')
df_N