import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

x,y = load_iris().data,load_iris().target

def modeling(optimizer='adam',base_node=4,activation='relu',batch_size=16):
    inputs = Input(shape = (x_train.shape[1]))
    x = Dense(base_node*128,activation='relu')(inputs)
    x = Dense(base_node*64,activation='relu')(x)
    x = Dense(base_node*32,activation='relu')(x)
    x = Dense(base_node*16,activation='relu')(x)
    x = Dense(base_node*8,activation='relu')(x)
    x = Dense(base_node*4,activation='relu')(x)
    outputs = Dense(3,activation='softmax')(x)
    model = Model(inputs=inputs,outputs=outputs)
    model.compile(loss = 'categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    return model

def params():
    base_node = [1,2,4,8]
    optimizer = ['adam','nadam']
    activation = ['relu','tanh','selu','elu']
    batch_size = [8,16,32]
    return {'optimizer':optimizer,'activation':activation,'batch_size':batch_size,'base_node':base_node}

def callbacks():
    filepath = "../data/modelcheckpoint/keras62_2_iris_{epoch:02d}-{val_accuracy:.f4}.hdf5"
    es = EarlyStopping(monitor = 'val_loss',patience=10)
    cp = ModelCheckpoint(monitor = 'val_loss',filepath = filepath,save_best_only=True)
    reLR = ReduceLROnPlateau(monitor = 'val_loss',patience=5)
    return es,cp,reLR
ES,CP,reLR = callbacks()

y = to_categorical(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
scaler = MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = params()

model = KerasClassifier(build_fn=modeling,verbose=1)
search = RandomizedSearchCV(model,parameters,cv=5)
search.fit(x_train,y_train,validation_split=0.2,epochs=100)#,callbacks=[ES,CP,reLR])

acc = search.score(x_test,y_test)
print(search.best_params_)
print(search.best_estimator_)
print(search.score)
print('최종 스코어 : ',acc)

'''
{'optimizer': 'adam', 'batch_size': 8, 'base_node': 2, 'activation': 'tanh'}
<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000016CCAFA8A60>
<bound method BaseSearchCV.score of RandomizedSearchCV(cv=5,
                   estimator=<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000016CD5D7ED60>,
                   param_distributions={'activation': ['relu', 'tanh', 'selu',
                                                       'elu'],
                                        'base_node': [1, 2, 4, 8],
                                        'batch_size': [8, 16, 32],
                                        'optimizer': ['adam', 'nadam']})>
최종 스코어 :  0.9333333373069763
'''