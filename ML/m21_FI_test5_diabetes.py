from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 데이터
datasets = load_diabetes()
x_train,x_test,y_train,y_test = train_test_split(datasets.data,datasets.target,test_size=0.15)

# 모델
model = RandomForestRegressor()

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
    print(len(feature_importances))
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
df.drop(cut_columns(model.feature_importances_,datasets.feature_names,3),axis=1,inplace=True)
print(cut_columns(model.feature_importances_,datasets.feature_names,3))
x_train,x_test,y_train,y_test = train_test_split(df.values,datasets.target,test_size=0.15)

model = RandomForestRegressor()

# 훈련
model.fit(x_train,y_train)
y = datasets.target
# 평가, 예측
acc = model.score(x_test,y_test)
print("acc : ",acc)

"""
['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
acc :  0.491502874922763
10
10
['sex', 's4', 's1']
acc :  0.4513483583133965
"""




