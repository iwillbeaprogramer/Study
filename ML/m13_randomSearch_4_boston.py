import datetime
start_time = datetime.datetime.now()
params = [
    {'n_estimators':[100,200],'max_depth':[6,8,10,12],'min_samples_leaf':[3,5,7,10],'min_samples_split':[2,3,5,10],'n_jobs':[-1]},
    #{'max_depth':[6,8,10,12]},
    #{'min_samples_leaf':[3,5,7,10]},
    #{'min_sample_split':[2,3,5,10]},
    #{'n_jobs':[-1]}
]
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,KFold,RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston

datasets = load_boston()
x = datasets.data
y = datasets.target
scaler = MinMaxScaler()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

kfold = KFold(n_splits=10,shuffle=True)

model = RandomizedSearchCV(RandomForestRegressor(),params,cv=kfold,)
model.fit(x_train,y_train)
print("최적의 매개변수 : ",model.best_estimator_)
y_pred = model.predict(x_test)
print('최종정답률 : ',model.score(x_test,y_test))
end_time = datetime.datetime.now()
print('걸린시간 : ',end_time-start_time)
'''
최적의 매개변수 :  RandomForestRegressor(max_depth=12, min_samples_leaf=3, n_jobs=-1)
최종정답률 :  0.8713646465019551
걸린시간 :  0:00:11.912686

'''
