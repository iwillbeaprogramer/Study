from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score,accuracy_score
import numpy as np

x,y = load_boston(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

model = XGBRegressor(n_jobs=8)
model.fit(x_train,y_train)

score = model.score(x_test,y_test)
print("R2 : ",score)

thresholds = np.sort(model.feature_importances_)
print(thresholds)

# [0.0014197  0.00346099 0.00571491 0.0102198  0.01241846 0.01352226
#  0.02160533 0.0337475  0.03405323 0.04388783 0.07190961 0.3375697
#  0.4104706 ]
for thresh in thresholds:
    selection = SelectFromModel(model,prefit=True,threshold=thresh)
    select_x_train = selection.transform(x_train)
    print(select_x_train.shape)
    selection_model = XGBRegressor(n_jobs=8)
    selection_model.fit(select_x_train,y_train)
    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)
    score=r2_score(y_test,y_predict)
    print("Thresh=%.3f,     n=%d,       R2:%.2F%%"%(thresh,select_x_train.shape[1],score*100))



print(model.coef_,model.intercept_)