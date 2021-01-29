from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV,KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

params = [
    {"model__C":[1,10,100,1000],"model__kernel":['sigmoid','linear','rbf'],"model__gamma":[0.1,0.01,0.001,0.0001]}
]
kfold = KFold(n_splits=5,shuffle=True)

datasets = load_iris()
x = datasets.data
y = datasets.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15)

pipe = Pipeline([('scaler',MinMaxScaler()),('model',SVC())])
model = GridSearchCV(pipe,params,cv=kfold)
model.fit(x_train,y_train)


print('model_score : ',model.score(x_test,y_test))
'''
pipeline score :  0.9565217391304348
pipeline score :  0.9565217391304348
'''

# pipeline(scaler,model) -> grid.fit