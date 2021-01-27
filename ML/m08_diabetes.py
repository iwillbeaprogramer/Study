from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score,r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC,SVC,SVR,LinearSVR
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor

datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model_list=[]

model1 = LinearSVR()
model2 = SVR()
model3 = KNeighborsRegressor()
model4 = LinearRegression()
model5 = DecisionTreeRegressor()
model6 = RandomForestRegressor()
model_list.append(model1)
model_list.append(model2)
model_list.append(model3)
model_list.append(model4)
model_list.append(model5)
model_list.append(model6)

for k,i in enumerate(model_list):
    i.fit(x_train,y_train)
    score = i.score(x_test,y_test)
    y_pred=i.predict(x_test)
    if k==0:
        print("LinearSVC Accuracy : ", score,"acc_score : ",r2_score(y_test,y_pred))
    elif k==1:
        print("SVC Accuracy : ", score,"acc_score : ",r2_score(y_test,y_pred))
    elif k==2:
        print("KNeighbors Accuracy : ", score,"acc_score : ",r2_score(y_test,y_pred))
    elif k==3:
        print("LinearRegressor Accuracy : ", score,"acc_score : ",r2_score(y_test,y_pred))
    elif k==4:
        print("DecisionTree Accuracy : ", score,"acc_score : ",r2_score(y_test,y_pred))
    else:
        print("RandomForest Accuracy : ", score,"acc_score : ",r2_score(y_test,y_pred))


'''
LinearSVC Accuracy :  0.11926306753175764 acc_score :  0.11926306753175764
SVC Accuracy :  0.07461559866525336 acc_score :  0.07461559866525336
KNeighbors Accuracy :  0.42042058689414963 acc_score :  0.42042058689414963
LinearRegressor Accuracy :  0.4689589434529088 acc_score :  0.4689589434529088
DecisionTree Accuracy :  -0.1709927607353865 acc_score :  -0.1709927607353865
RandomForest Accuracy :  0.36425946893895866 acc_score :  0.36425946893895866
Tensorflow Accuracy : 0.21290600730061382
'''