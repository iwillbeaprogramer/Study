import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

x,y = load_breast_cancer().data,load_breast_cancer().target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

def modeling(batch_size=32,optimizer='adam',activation='relu',base_node=4):
    inputs = Input(shape=(30,))
    x = Dense(base_node*256,activation=activation)(inputs)
    x = Dense(base_node*128,activation=activation)(x)
    x = Dense(base_node*64,activation=activation)(x)
    x = Dense(base_node*32,activation=activation)(x)
    x = Dense(base_node*16,activation=activation)(x)
    x = Dense(base_node*4,activation=activation)(x)
    outputs = Dense(1,activation='sigmoid')(x)
    model = Model(inputs=inputs,outputs=outputs)
    model.compile(loss = 'binary_crossentropy',optimizer = optimizer,metrics = ['accuracy'])
    return model

def make_params():
    batch_size=[4,8,16,32]
    activation =['relu','selu','tanh']
    base_node = [1,2,4,8]
    return {"batch_size":batch_size,"activation":activation,'base_node':base_node}



params = make_params()
model = KerasClassifier(build_fn = modeling,verbose=1)
search = RandomizedSearchCV(model,params,cv=5)
search.fit(x_train,y_train,epochs=100,validation_split=0.2)

acc = search.score(x_test,y_test)
print(search.best_params_)
print(search.best_estimator_)
print(search.score)
print('최종 스코어 : ',acc)

'''
{'batch_size': 4, 'base_node': 1, 'activation': 'selu'}
<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x00000219F37A90A0>
<bound method BaseSearchCV.score of RandomizedSearchCV(cv=5,
                   estimator=<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000002189BF71B50>,
                   param_distributions={'activation': ['relu', 'selu', 'tanh'],
                                        'base_node': [1, 2, 4, 8],
                                        'batch_size': [4, 8, 16, 32]})>
'''