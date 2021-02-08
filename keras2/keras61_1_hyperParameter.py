import numpy as np
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Dropout,Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

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

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model,verbose=1)


params = create_hyperparameters()
# model2 = build_model()
from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(model2, params,cv=3)
search.fit(x_train,y_train,verbose=1)

acc = search.score(x_test,y_test)
print(search.best_params_)
print(search.best_estimator_)
print(search.score)
print("최종스코어 : ",acc)


'''

{'optimizer': 'adam', 'drop': 0.1, 'batch_size': 20}
<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001E75C7909A0>
<bound method BaseSearchCV.score of RandomizedSearchCV(cv=3,
                   estimator=<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001E748507D30>,
                   param_distributions={'batch_size': [10, 20, 30, 40, 50],
                                        'drop': [0.1, 0.2, 0.3],
                                        'optimizer': ['rmsprop', 'adam',
                                                      'adadelta']})>
최종스코어 :  0.9672999978065491
'''