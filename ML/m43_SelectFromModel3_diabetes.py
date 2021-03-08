# 실습
# 상단 모델에 그리드서치 또는 랜덤서치로 튜닝한 모델 구성
# 최적의 r2값과 피처임포턴스 구할것

#2 위 쓰레드 값으로 selectfrommodel을 구해서 최적의 피쳐 갯수를 구할것

#3 위 피쳐 갯수로 데이터 (피쳐)를 수정(삭제) 해서 그리드 서치 또는 랜덤서치 적용하여 최적의 r2값 구할것

#1 번값과 2번값 비교

from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split,GridSearchCV,KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score,accuracy_score
import numpy as np

params = [
    {'n_estimators':[100,200],'max_depth':[6,8,10,12],'min_samples_leaf':[3,5,7,10],'min_samples_split':[2,3,5,10],'n_jobs':[-1]},
    #{'max_depth':[6,8,10,12]},
    #{'min_samples_leaf':[3,5,7,10]},
    #{'min_sample_split':[2,3,5,10]},
    #{'n_jobs':[-1]}
]

kfold = KFold(n_splits=4,shuffle=True,random_state=0)

x,y = load_diabetes(return_X_y=True)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
# xgbr = XGBRegressor()

# model = GridSearchCV(XGBRegressor(),params,cv=kfold,verbose=True)
# model.fit(x_train,y_train)
# score = model.score(x_test,y_test)
# print("최적의 매개변수 : ",model.best_estimator_)
# print('최종정답률 : ',model.score(x_test,y_test))

# 최적의 파라미터
# base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.300000012, max_delta_step=0, max_depth=8,
#              min_child_weight=1, min_samples_leaf=3, min_samples_split=2,
#              missing=nan, monotone_constraints='()', n_estimators=100,
#              n_jobs=-1, num_parallel_tree=1, random_state=0, reg_alpha=0,
#              reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
#              validate_parameters=1, verbosity=None

# 최적의 r2
# 0.8985656375675573


model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.300000012, max_delta_step=0, max_depth=8,
             min_child_weight=1, min_samples_leaf=3, min_samples_split=2,
             monotone_constraints='()', n_estimators=100,
             n_jobs=-1, num_parallel_tree=1, random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)

model.fit(x_train,y_train)

score = model.score(x_test,y_test)
print("R2 : ",score)

thresholds = np.sort(model.feature_importances_)
print(thresholds)


# 최적의 feature_importances
# [0.00101464 0.0060258  0.00617483 0.00857513 0.01081747 0.01440269
#  0.0213401  0.03397179 0.0357166  0.03916002 0.04662089 0.18326165
#  0.5929184 ]

count=0

for thresh in thresholds:
    selection = SelectFromModel(model,prefit=True,threshold=thresh)
    select_x_train = selection.transform(x_train)
    print(select_x_train.shape)
    selection_model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.300000012, max_delta_step=0, max_depth=8,
             min_child_weight=1,  min_samples_split=2,
             monotone_constraints='()', n_estimators=100,
             n_jobs=-1, num_parallel_tree=1, random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
    selection_model.fit(select_x_train,y_train)
    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)
    score=r2_score(y_test,y_predict)
    print("Thresh=%.3f,     n=%d,       R2:%.2F%%"%(thresh,select_x_train.shape[1],score*100))
    if count==0:
            best_score=score
            best_index=count
            best_select = selection
            best_select_x_train = select_x_train
            best_select_x_test = select_x_test
    else:
        if best_score<score:
            best_score=score
            best_index=count
            best_select = selection
            best_select_x_train = select_x_train
            best_select_x_test = select_x_test
    count+=1
print("최고점수 : ",best_score*100,"    최고 인덱스 : ",best_index)
model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.300000012, max_delta_step=0, max_depth=8,
             min_child_weight=1,  min_samples_split=2,
             monotone_constraints='()', n_estimators=100,
             n_jobs=-1, num_parallel_tree=1, random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)

model.fit(best_select_x_train,y_train)
score = model.score(best_select_x_test,y_test)
model = GridSearchCV(XGBRegressor(),params,cv=kfold,verbose=True)

model.fit(x_train,y_train)
print("최적의 매개변수 : ",model.best_estimator_)
y_pred = model.predict(x_test)
print('최종정답률 : ',model.score(x_test,y_test))




# Thresh=0.002,     n=13,       R2:91.75%
# (404, 12)
# Thresh=0.005,     n=12,       R2:91.92%
# (404, 11)
# Thresh=0.007,     n=11,       R2:91.70%
# (404, 10)
# Thresh=0.008,     n=10,       R2:91.69%
# (404, 9)
# Thresh=0.008,     n=9,       R2:91.41%
# (404, 8)
# Thresh=0.012,     n=8,       R2:92.30%
# (404, 7)
# Thresh=0.020,     n=7,       R2:90.86%
# (404, 6)
# Thresh=0.033,     n=6,       R2:91.77%
# (404, 5)
# Thresh=0.037,     n=5,       R2:92.37%
# (404, 4)
# Thresh=0.050,     n=4,       R2:91.47%
# (404, 3)
# Thresh=0.058,     n=3,       R2:88.05%
# (404, 2)
# Thresh=0.170,     n=2,       R2:77.87%
# (404, 1)
# Thresh=0.590,     n=1,       R2:48.92%

# 최적의 피쳐갯수 12


"""

최적의 매개변수 :  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.300000012, max_delta_step=0, max_depth=10,
             min_child_weight=1, min_samples_leaf=3, min_samples_split=2,
             missing=nan, monotone_constraints='()', n_estimators=100,
             n_jobs=-1, num_parallel_tree=1, random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
최종정답률 :  0.5148071903933924

"""