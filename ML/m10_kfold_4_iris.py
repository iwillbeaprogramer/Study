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

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=77,shuffle=True)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



model_list=[]
model1 = LinearSVC()
model2 = SVC()
model3 = KNeighborsClassifier()
model4 = LogisticRegression()
model5 = DecisionTreeClassifier()
model6 = RandomForestClassifier()
model_list.append(model1)
model_list.append(model2)
model_list.append(model3)
model_list.append(model4)
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
        print("LogisticRegressor Accuracy : \n", score)
    elif k==4:
        print("DecisionTree Accuracy : \n", score)
    else:
        print("RandomForest Accuracy : \n", score)

'''
LinearSVC Accuracy : 
 [0.83333333 0.91666667 0.875      0.95833333 0.91666667]
SVC Accuracy :
 [1.         0.95833333 1.         1.         0.95833333]
KNeighbors Accuracy :
 [1.         0.95833333 0.95833333 0.95833333 1.        ]
LogisticRegressor Accuracy : 
 [0.875      0.875      0.91666667 0.91666667 0.875     ]
DecisionTree Accuracy :
 [1.         1.         1.         0.95833333 0.875     ]
RandomForest Accuracy : 
 [0.95833333 0.95833333 0.95833333 1.         1.        ]
'''




