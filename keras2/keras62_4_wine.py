import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.models import Model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

datasets = load_wine()
x,y = datasets.data,datasets.target
y = to_categorical(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

def modeling(batch_size=16,optimizer='adam',base_node=2,activation='relu'):
    inputs = Input(shape=(x_train.shape[1],))
    x = Dense(base_node*256,activation=activation)(inputs)
    x = Dense(base_node*128,activation=activation)(x)
    x = Dense(base_node*64,activation=activation)(x)
    x = Dense(base_node*32,activation=activation)(x)
    x = Dense(base_node*8,activation=activation)(x)
    x = Dense(base_node*4,activation=activation)(x)
    outputs = Dense(3,activation='softmax')(x)
    model = Model(inputs=inputs,outputs=outputs)
    model.compile(loss = 'categorical_crossentropy',metrics=['accuracy'],optimizer = optimizer)
    return model

def make_params():
    batch_size = [1,2,4,8,16]
    optimizer = ['adam','nadam']
    base_node = [1,2,4,8]
    activation = ['relu','selu','tanh','leaky_relu']
    return {'batch_size':batch_size,"optimizer":optimizer,"base_node":base_node,"activation":activation}

model = KerasClassifier(build_fn=modeling,verbose=1)
params = make_params()

search = RandomizedSearchCV(model,params,cv=5)
search.fit(x_train,y_train,epochs = 150,validation_split=0.2)
acc = search.score(x_test,y_test)
print(search.best_estimator_)
print(search.best_params_)
print(search.score)
print("최종 스코어 : ",acc)
'''
<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001A6B9DF0BB0>
{'optimizer': 'adam', 'batch_size': 2, 'base_node': 8, 'activation': 'selu'}
<bound method BaseSearchCV.score of RandomizedSearchCV(cv=5,
                   estimator=<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001A6BB154D90>,
                   param_distributions={'activation': ['relu', 'selu', 'tanh',
                                                       'leaky_relu'],
                                        'base_node': [1, 2, 4, 8],
                                        'batch_size': [1, 2, 4, 8, 16],
                                        'optimizer': ['adam', 'nadam']})>
최종 스코어 :  0.9722222089767456
'''