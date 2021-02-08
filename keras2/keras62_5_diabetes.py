import numpy as np
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Dropout,Input
from sklearn.datasets import load_diabetes
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
datasets = load_diabetes()
x,y = datasets.data,datasets.target

# 1. 데이터 전처리
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


def build_model(drop=0.5,optimizer='adam'):
    inputs = Input(shape = (x_train.shape[1]),name = 'input')
    x = Dense(512,activation='relu',name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256,activation='relu',name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128,activation='relu',name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(1,name = 'outputs')(x)
    model = Model(inputs=inputs,outputs=outputs)
    model.compile(loss = 'mse',optimizer = optimizer,metrics = ['mse'])
    return model

def create_hyperparameters():
    batches = [10,20,30,40,50]
    optimizers = ['rmsprop','adam','adadelta']
    dropout = [0.1,0.2,0.3]
    # epochs = [100,300,500]
    return {'batch_size':batches,'optimizer':optimizers,'drop':dropout}

def callbacks():
    modelpath = "../data/modelcheckpoint/keras62_boston_{epoch:02d}-{val_loss:.4f}.hdf5"
    es = EarlyStopping(monitor = 'val_loss',patience=5)
    reLR = ReduceLROnPlateau(monitor = 'val_loss',patience=3,factor=0.8)
    cp = ModelCheckpoint(monitor = 'val_loss',mode='auto',filepath=modelpath)
    return es, reLR,cp

es,reLR,cp = callbacks()

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
model2 = KerasRegressor(build_fn=build_model)


params = create_hyperparameters()
# model2 = build_model()
from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(model2, params,cv=3)
search.fit(x_train,y_train,verbose=1,epochs=100,validation_split=0.2,callbacks=[es,reLR,cp])

acc = search.score(x_test,y_test)
print(search.best_params_)
print(search.best_estimator_)
print(search.score)
print("최종스코어 : ",acc)


'''
{'optimizer': 'rmsprop', 'drop': 0.2, 'batch_size': 40}
<tensorflow.python.keras.wrappers.scikit_learn.KerasRegressor object at 0x000001E63E3FE130>
<bound method BaseSearchCV.score of RandomizedSearchCV(cv=3,
                   estimator=<tensorflow.python.keras.wrappers.scikit_learn.KerasRegressor object at 0x000001E619F58550>,
                   param_distributions={'batch_size': [10, 20, 30, 40, 50],
                                        'drop': [0.1, 0.2, 0.3],
                                        'optimizer': ['rmsprop', 'adam',
                                                      'adadelta']})>
최종스코어 :  -2877.880126953125
'''