from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score

datasets = load_boston()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
model = XGBRegressor(n_estimators = 1000,learning_rate = 0.01,tree_method = 'gpu_hist',predictor = 'gpu_predictor',gpu_id = 0)
model.fit(x_train,y_train,verbose=1,eval_metric=['rmse'],eval_set = [(x_train,y_train),(x_test,y_test)],early_stopping_rounds=1000)

score = model.score(x_test,y_test)
print("model's score : ",score)
