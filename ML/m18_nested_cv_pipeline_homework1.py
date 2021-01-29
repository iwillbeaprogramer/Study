# 데이터 : diabetes
# 모델 : RandomForest
# 파이프라인 엮어서 25번 돌리기!!

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,StandardScaler

params = [
    {'model__n_estimators':[100,200,300,400],'model__max_depth':[6,8,10,12],'model__min_samples_leaf':[3,5,7,10],'model__min_samples_split':[2,3,5,7,10],'model__n_jobs':[-1]},
]

datasets = load_diabetes()
x = datasets.data
y = datasets.target
kfold = KFold(n_splits=5,shuffle=True)


for i in [MinMaxScaler,StandardScaler]:
    pipeline = Pipeline([('scaler',i()),('model',RandomForestRegressor())])
    model = GridSearchCV(pipeline,params,cv=kfold)
    score = cross_val_score(model,x,y,cv=kfold)
    print(i.__name__)
    print("scores : ",score)


'''
MinMaxScaler
scores :  [0.56970197 0.48346884 0.39402616 0.46015026 0.44574737]
StandardScaler
scores :  [0.52915697 0.35088479 0.44009741 0.36233404 0.46947654]
'''
###    cross_val_score(grid(pipeline(model)))