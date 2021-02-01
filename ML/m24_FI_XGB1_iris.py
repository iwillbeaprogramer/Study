from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
# 데이터
datasets = load_iris()
x_train,x_test,y_train,y_test = train_test_split(datasets.data,datasets.target,test_size=0.15)

# 모델
starttime = datetime.datetime.now()
model = XGBClassifier(n_jobs=-1)

# 훈련
model.fit(x_train,y_train)
y = datasets.target
# 평가, 예측
acc = model.score(x_test,y_test)
print(model.feature_importances_)
print(datasets.feature_names)
print("acc : ",acc)

def plot_feature_importances_datasets(model,datasets):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features),datasets.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel("Features")
    plt.ylim(-1,n_features)

plot_feature_importances_datasets(model,datasets)
#plt.show()


def cut_columns(feature_importances,columns,number):
    temp = []
    for i in feature_importances:
        temp.append(i)
    temp.sort()
    temp=temp[:number]
    result = []
    for j in temp:
        index = feature_importances.tolist().index(j)
        result.append(columns[index])
    return result

df = pd.DataFrame(datasets.data,columns=datasets.feature_names)
df.drop(cut_columns(model.feature_importances_,datasets.feature_names,1),axis=1,inplace=True)
print(cut_columns(model.feature_importances_,datasets.feature_names,1))
x_train,x_test,y_train,y_test = train_test_split(df.values,datasets.target,test_size=0.15)

model = XGBClassifier(n_jobs=-1)

# 훈련
model.fit(x_train,y_train)
y = datasets.target
# 평가, 예측
acc = model.score(x_test,y_test)
print("acc : ",acc)
endtime = datetime.datetime.now()
print('걸린사긴 : ',endtime-starttime)

'''
랜포
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
acc :  0.9565217391304348
['sepal width (cm)']
acc :  0.9130434782608695



그부
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
acc :  0.9565217391304348
['sepal length (cm)']
acc :  1.0

xgboost
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
acc :  1.0
['sepal length (cm)']
[10:05:45] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.3.0/src/learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'multi:softprob' was changed from 'merror' to 'mlogloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
acc :  1.0

-1 걸린사긴 :  0:00:00.332334

'''



