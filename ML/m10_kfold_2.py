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

scores = cross_val_score(model1,x_train,y_train,cv=kfold)
print("scores : ",scores)


for k,i in enumerate(model_list):
    i.fit(x_train,y_train)
    score = i.score(x_test,y_test)
    if k==0:
        print("LinearSVC Accuracy : ", score)
    elif k==1:
        print("SVC Accuracy : ", score)
    elif k==2:
        print("KNeighbors Accuracy : ", score)
    elif k==3:
        print("LogisticRegressor Accuracy : ", score)
    elif k==4:
        print("DecisionTree Accuracy : ", score)
    else:
        print("RandomForest Accuracy : ", score)

'''
LinearSVC Accuracy :  0.9333333333333333
SVC Accuracy :  0.9333333333333333
KNeighbors Accuracy :  0.9333333333333333
LogisticRegressor Accuracy :  0.9333333333333333
DecisionTree Accuracy :  0.9
RandomForest Accuracy :  0.9333333333333333
Tensorflow Accuracy : 0.9666666388511658
'''




