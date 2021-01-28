import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,LabelEncoder
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split,KFold,cross_val_score

from sklearn.svm import LinearSVC,SVC,LinearSVR,SVR
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor




datasets = load_diabetes()
x = datasets.data
y = datasets.target
# x,y = load_iris(return_X_y=True)
kfold = KFold(n_splits=5,shuffle=True)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=77,shuffle=True)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



model_list=[]
model1 = LinearSVR()
model2 = SVR()
model3 = KNeighborsRegressor()
model5 = DecisionTreeRegressor()
model6 = RandomForestRegressor()
model_list.append(model1)
model_list.append(model2)
model_list.append(model3)
model_list.append(model5)
model_list.append(model6)





for k,i in enumerate(model_list):
    score = cross_val_score(i,x_train,y_train,cv=kfold)
    i.fit(x_train,y_train)
    if k==0:
        print("LinearSVC Accuracy : \n", score)
    elif k==1:
        print("SVC Accuracy : \n", score)
    elif k==2:
        print("KNeighbors Accuracy : \n", score)
    elif k==3:
        print("DecisionTree Accuracy : \n", score)
    else:
        print("RandomForest Accuracy : \n", score)

'''
LinearSVC Accuracy : 
 [0.20842892 0.15952474 0.07468538 0.16363503 0.21473862]
SVC Accuracy :
 [0.12688912 0.0502222  0.08796397 0.14625835 0.12632862]
KNeighbors Accuracy : 
 [0.30988968 0.20347932 0.24369371 0.42063774 0.53454774]
DecisionTree Accuracy : 
 [ 0.03469526 -0.06124415  0.08071419 -0.38895618 -0.2623582 ]
RandomForest Accuracy : 
 [0.29553591 0.47465525 0.47661283 0.49894092 0.3570791 ]
'''




