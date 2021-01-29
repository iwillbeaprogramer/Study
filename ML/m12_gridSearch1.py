import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,LabelEncoder
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV

from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')



datasets = load_iris()
x = datasets.data
y = datasets.target
# x,y = load_iris(return_X_y=True)
kfold = KFold(n_splits=5,shuffle=True)


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

params = [
    {"C":[1,10,100,1000],'kernel':['linear']},
    {"C":[1,10,100],'kernel':['rbf'],'gamma':[0.001,0.0001]},
    {"C":[1,10,100,1000],'kernel':['sigmoid'],"gamma":[0.001,0.0001]}
]
model = GridSearchCV(SVC(),params,cv=kfold)

model.fit(x_train,y_train)
print("최적의 매개변수 : ",model.best_estimator_)
y_pred = model.predict(x_test)
print('최종정답률 : ',model.score(x_test,y_test))







