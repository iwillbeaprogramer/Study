import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,LabelEncoder
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split,KFold,cross_val_score

from sklearn.svm import LinearSVC,SVC,LinearSVR,SVR
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor




datasets = load_boston()
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
 [0.28942464 0.65192463 0.66745029 0.70031067 0.7302328 ]
SVC Accuracy : 
 [0.56894709 0.68776758 0.52438705 0.70461139 0.42917993]
KNeighbors Accuracy : 
 [0.60564868 0.70903878 0.73521573 0.63939482 0.79361249]
DecisionTree Accuracy : 
 [0.85521254 0.73033114 0.75940516 0.51582054 0.79175947]
RandomForest Accuracy : 
 [0.92778725 0.74067769 0.85900759 0.78400173 0.88061792]
'''




