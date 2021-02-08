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
print(search.cv_results_)


'''

{'mean_fit_time': array([ 4.56517069,  4.47317767,  6.29844991,  4.58575749, 15.74045769,
        4.38161731,  3.57756829,  4.5473249 ,  5.44018332, 15.69242986]), 'std_fit_time': array([1.16090341, 0.1491078 , 0.16733113, 0.07048308, 0.74008593,
       0.10406496, 0.03936128, 0.08245841, 0.07531798, 0.97922022]), 'mean_score_time': array([1.19885929, 1.54212928, 1.76392579, 1.55105162, 3.27918514,
       1.40474455, 1.01269452, 1.39268883, 1.66216731, 3.94213398]), 'std_score_time': array([0.0586614 , 0.1065254 , 0.06276998, 0.06605173, 0.14581825,
       0.01496619, 0.02051807, 0.00632922, 0.03011918, 0.61983828]), 'param_optimizer': masked_array(data=['adam', 'adam', 'adadelta', 'adam', 'rmsprop', 'adam',
                   'adadelta', 'adadelta', 'adam', 'rmsprop'],
             mask=[False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'param_drop': masked_array(data=[0.1, 0.2, 0.1, 0.1, 0.3, 0.3, 0.2, 0.1, 0.1, 0.1],
             mask=[False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'param_batch_size': masked_array(data=[40, 30, 20, 30, 10, 30, 40, 30, 20, 10],
             mask=[False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'params': [{'optimizer': 'adam', 'drop': 0.1, 'batch_size': 40}, {'optimizer': 'adam', 'drop': 0.2, 'batch_size': 30}, {'optimizer': 'adadelta', 'drop': 0.1, 'batch_size': 20}, {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 30}, {'optimizer': 'rmsprop', 'drop': 0.3, 'batch_size': 10}, {'optimizer': 'adam', 'drop': 0.3, 'batch_size': 30}, {'optimizer': 'adadelta', 'drop': 0.2, 'batch_size': 40}, {'optimizer': 'adadelta', 'drop': 0.1, 'batch_size': 30}, {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 20}, {'optimizer': 'rmsprop', 'drop': 0.1, 'batch_size': 10}], 'split0_test_score': array([0.96125001, 0.96195   , 0.33184999, 0.96139997, 0.94515002,
       0.95775002, 0.22775   , 0.33074999, 0.95169997, 0.95485002]), 'split1_test_score': array([0.95074999, 0.95370001, 0.37445   , 0.95389998, 0.94195002,
       0.95025003, 0.1684    , 0.25650001, 0.95550001, 0.94594997]), 'split2_test_score': array([0.95784998, 0.95770001, 0.35304999, 0.95490003, 0.95249999,
       0.95174998, 0.21335   , 0.35714999, 0.95770001, 0.9576    ]), 'mean_test_score': array([0.95661666, 0.95778334, 0.35311666, 0.95673333, 0.94653334,
       0.95325001, 0.20316667, 0.31479999, 0.95496666, 0.9528    ]), 'std_test_score': array([0.00437442, 0.00336856, 0.01739144, 0.00332498, 0.00441668,
'''