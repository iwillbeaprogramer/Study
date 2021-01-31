from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 데이터
datasets = load_breast_cancer()
x_train,x_test,y_train,y_test = train_test_split(datasets.data,datasets.target,test_size=0.15)

# 모델
model = RandomForestClassifier()

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
df.drop(cut_columns(model.feature_importances_,datasets.feature_names,8),axis=1,inplace=True)
print(cut_columns(model.feature_importances_,datasets.feature_names,8))
x_train,x_test,y_train,y_test = train_test_split(df.values,datasets.target,test_size=0.15)

model = RandomForestClassifier()

# 훈련
model.fit(x_train,y_train)
y = datasets.target
# 평가, 예측
acc = model.score(x_test,y_test)
print("acc : ",acc)


"""
랜포
acc :  0.9534883720930233



빼고나서
['mean symmetry', 'mean fractal dimension', 'symmetry error', 'texture error', 'smoothness error', 'compactness error', 'fractal dimension error', 'mean compactness']
acc :  0.9418604651162791
"""


