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

params = [
    {"C":[1,10,100,1000],'kernel':['linear']},
    {"C":[1,10,100],'kernel':['rbf'],'gamma':[0.001,0.0001]},
    {"C":[1,10,100,1000],'kernel':['sigmoid'],"gamma":[0.001,0.0001]}
]

model = GridSearchCV(SVC(),params,cv=kfold)
score = cross_val_score(model,x_train,y_train,cv=kfold)

print("교차검증점수 : ",score)







