from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score,r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC,SVC,SVR,LinearSVR
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
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
    i.fit(x_train,y_train)
    score = i.score(x_test,y_test)
    y_pred=i.predict(x_test)
    if k==0:
        print("LinearSVC Accuracy : ", score,"acc_score : ",accuracy_score(y_test,y_pred))
    elif k==1:
        print("SVC Accuracy : ", score,"acc_score : ",accuracy_score(y_test,y_pred))
    elif k==2:
        print("KNeighbors Accuracy : ", score,"acc_score : ",accuracy_score(y_test,y_pred))
    elif k==3:
        print("Logistic Accuracy : ", score,"acc_score : ",accuracy_score(y_test,y_pred))
    elif k==4:
        print("DecisionTree Accuracy : ", score,"acc_score : ",accuracy_score(y_test,y_pred))
    else:
        print("RandomForest Accuracy : ", score,"acc_score : ",accuracy_score(y_test,y_pred))

'''
LinearSVC Accuracy :  0.9649122807017544 acc_score :  0.9649122807017544
SVC Accuracy :  0.9649122807017544 acc_score :  0.9649122807017544
KNeighbors Accuracy :  0.9473684210526315 acc_score :  0.9473684210526315
Logistic Accuracy :  0.9385964912280702 acc_score :  0.9385964912280702
DecisionTree Accuracy :  0.9298245614035088 acc_score :  0.9298245614035088
RandomForest Accuracy :  0.9649122807017544 acc_score :  0.9649122807017544
Tensorflow Accuracy : 0.9824561476707458
'''