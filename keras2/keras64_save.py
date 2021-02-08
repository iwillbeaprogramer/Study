# 가중치 저장할것
#1. model.save()
#2. pickle쓸것


import numpy as np
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Dropout,Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau

(x_train,y_train),(x_test,y_test) = mnist.load_data()
# 1. 데이터 전처리

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(-1,28*28).astype('float32')/255.0
x_test = x_test.reshape(-1,28*28).astype('float32')/255.0


def build_model(drop=0.5,optimizer='adam'):
    inputs = Input(shape = (28*28),name = 'input')
    x = Dense(512,activation='relu',name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256,activation='relu',name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128,activation='relu',name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10,activation='softmax',name = 'outputs')(x)
    model = Model(inputs=inputs,outputs=outputs)
    model.compile(loss = 'categorical_crossentropy',optimizer = optimizer,metrics = ['acc'])
    return model

def create_hyperparameters():
    batches = [10,20,30,40,50]
    optimizers = ['rmsprop','adam','adadelta']
    dropout = [0.1,0.2,0.3]
    return {'batch_size':batches,'optimizer':optimizers,'drop':dropout}

def callbacks():
    modelpath = "../data/modelcheckpoint/keras64_save.hdf5"
    es = EarlyStopping(monitor = 'val_loss',patience=10)
    cp = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath,save_best_only=True)
    reLR = ReduceLROnPlateau(monitor = 'val_loss',patience=5)
    return es,reLR,cp

es,reLR,cp = callbacks()


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model,verbose=1)


params = create_hyperparameters()
# model2 = build_model()
from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(model2, params,cv=3)
search.fit(x_train,y_train,verbose=1,epochs=1,callbacks=[es,reLR,cp])

import pickle
pickle.dump(model2, open('../data/modelcheckpoint/keras64.pickle.dat', 'wb'))
model4 = pickle.load(open('../data/modelcheckpoint/keras64.pickle.dat', 'rb'))



acc = search.score(x_test,y_test)
search.best_estimator_.model.save('../data/modelcheckpoint/k64_Grid_model_save.h5') 
print(search.best_params_)
print(search.best_estimator_)
print(search.score)
print("최종스코어 : ",acc)


'''

{'optimizer': 'adam', 'drop': 0.1, 'batch_size': 30}
<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001A44E8AAA00>
<bound method BaseSearchCV.score of RandomizedSearchCV(cv=3,
                   estimator=<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001A439788550>,
                   param_distributions={'batch_size': [10, 20, 30, 40, 50],
                                        'drop': [0.1, 0.2, 0.3],
                                        'optimizer': ['rmsprop', 'adam',
                                                      'adadelta']})>
최종스코어 :  0.9666000008583069
'''