# 실습 or 과제
#train,test 나눈다음에 train만 발리데이션 하지말고,
#kfold한 후에 train_test_split사용

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,LabelEncoder
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split,KFold,cross_val_score

from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier




datasets = load_iris()
x = datasets.data
y = datasets.target
# x,y = load_iris(return_X_y=True)
kfold = KFold(n_splits=5,shuffle=True)



model1 = LinearSVC()

for train_index, test_index in kfold.split(x):
#    print('================================================================================')
#    print("TRAIN:", train_index, "\nTEST:", test_index) 

    # train : test
    x_train, x_test = x[train_index], x[test_index] 
    y_train, y_test = y[train_index], y[test_index]
      
    # train : test : validation
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = 77, shuffle = False) 

#    print('x_train.shape : ', x_train.shape)        # (96, 4)
#    print('x_test.shape  : ', x_test.shape)         # (30, 4)
#    print('x_val.shape   : ', x_val.shape)          # (24, 4)

#    print('y_train.shape : ', y_train.shape)        # (96, )
#    print('y_test.shape  : ', y_test.shape)         # (30, )
#    print('y_val.shape   : ', y_val.shape)          # (24, )

    # print('x_train.shape : \n', x_train)              
    # print('x_test.shape  : \n', x_test)               
    # print('x_val.shape   : \n', x_val)   

    # print('y_train.shape : \n', y_train)              
    # print('y_test.shape  : \n', y_test)
    # print('y_val.shape   : \n', y_val)

    scores = cross_val_score(model1,x_train,y_train,cv=kfold)
    print("scores : ",scores)

