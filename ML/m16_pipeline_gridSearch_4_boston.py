from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.model_selection import train_test_split,GridSearchCV,KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC

params = [
    {'model__n_estimators':[100,200,300,400],'model__max_depth':[6,8,10,12],'model__min_samples_leaf':[3,5,7,10],'model__min_samples_split':[2,3,5,7,10],'model__n_jobs':[-1]},
]

# params = [
#     {'randomforest__n_estimators':[100,200,300,400],'randomforest__max_depth':[6,8,10,12],'randomforest__min_samples_leaf':[3,5,7,10],'randomforest__min_samples_split':[2,3,5,7,10],'randomforest__n_jobs':[-1]},
# ]
kfold = KFold(n_splits=5,shuffle=True)
datasets = load_boston()
x = datasets.data
y = datasets.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15)

for i in [MinMaxScaler,StandardScaler]:

    pipe = Pipeline([('scaler',i()),('model',RandomForestRegressor())])
    # pipe = make_pipeline([MinMaxScaler(),RandomForestClassifier])
    model = GridSearchCV(pipe,params,cv=kfold)
    model.fit(x_train,y_train)

    print(i.__name__)
    print('best : ',model.best_estimator_)
    print('model_score : ',model.score(x_test,y_test))

'''
MinMaxScaler
best :  Pipeline(steps=[('scaler', MinMaxScaler()),
                ('model',
                 RandomForestRegressor(max_depth=10, min_samples_leaf=3,
                                       min_samples_split=3, n_estimators=300,
                                       n_jobs=-1))])
model_score :  0.8425645473111709
StandardScaler
best :  Pipeline(steps=[('scaler', StandardScaler()),
                ('model',
                 RandomForestRegressor(max_depth=6, min_samples_leaf=3,
                                       min_samples_split=3, n_jobs=-1))])
model_score :  0.8279111004701112
'''

# pipeline(scaler,model) -> grid.fit