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


model = XGBRegressor(n_estimators = 1000,learning_rate = 0.01,n_jobs=8)


model.fit(x_train,y_train,verbose=1,eval_metric = ['rmse','logloss'],eval_set = [(x_train,y_train),(x_test,y_test)],early_stopping_rounds=10)
aaa = model.score(x_test,y_test)
print(aaa)
y_pred = model.predict(x_test)
r2 = r2_score(y_test,y_pred)
rmse = mean_squared_error(y_test,y_pred)**0.5
print("r2 : ",r2)
print("rmse : ",rmse)

result = model.evals_result()
print(result)

import matplotlib.pyplot as plt
epochs = len(result['validation_0']['logloss'])
x_axis = range(0,epochs)

fig,ax = plt.subplots()
ax.plot(x_axis,result['validation_0']['logloss'],label='Train')
ax.plot(x_axis,result['validation_1']['logloss'],label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')

fig,ax = plt.subplots()
ax.plot(x_axis,result['validation_0']['rmse'],label='Train')
ax.plot(x_axis,result['validation_1']['rmse'],label='Test')
ax.legend()
plt.ylabel('Rmse')
plt.title('XGBoost RMSE')
plt.show()

