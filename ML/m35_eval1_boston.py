# eval_set

from xgboost import XGBClassifier,XGBRegressor
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score,accuracy_score,mean_squared_error

datasets = load_boston()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,shuffle=True,random_state = 66)


model = XGBRegressor(n_estimators = 500,learning_rate = 0.01,n_jobs=8)


model.fit(x_train,y_train,verbose=1,eval_metric = 'rmse',eval_set = [(x_train,y_train),(x_test,y_test)])
aaa = model.score(x_test,y_test)
print(aaa)
y_pred = model.predict(x_test)
r2 = r2_score(y_test,y_pred)
rmse = mean_squared_error(y_test,y_pred)**0.5
print("r2 : ",r2)
print("rmse : ",rmse)

result = model.evals_result()
print(result)
