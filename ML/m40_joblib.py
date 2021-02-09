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

# 저장
import pickle
import joblib
# pickle.dump(model, open("../data/xgb_save/m39.pickle.dat","wb"))
# print("저장완료")

# print("불러오기")
# model2 = pickle.load(open('../data/xgb_save/m39.pickle.dat','rb'))
# print('불러왔다')
# r22 = model.score(x_test,y_test)
# print('r22 : ',r22)
joblib.dump(model,"../data/xgb_save/m39.jopblib.dat")
model2 = joblib.load('../data/xgb_save/m39.jopblib.dat')