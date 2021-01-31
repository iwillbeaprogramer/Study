from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 데이터
datasets = load_wine()
x_train,x_test,y_train,y_test = train_test_split(datasets.data,datasets.target,test_size=0.15)

# 모델
model = GradientBoostingClassifier()

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
df.drop(cut_columns(model.feature_importances_,datasets.feature_names,4),axis=1,inplace=True)
print(cut_columns(model.feature_importances_,datasets.feature_names,4))
x_train,x_test,y_train,y_test = train_test_split(df.values,datasets.target,test_size=0.15)

model = GradientBoostingClassifier()

# 훈련
model.fit(x_train,y_train)
y = datasets.target
# 평가, 예측
acc = model.score(x_test,y_test)
print("acc : ",acc)

'''
랜포
['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
acc :  1.0
13
13
['nonflavanoid_phenols', 'ash', 'proanthocyanins', 'magnesium']
acc :  0.9259259259259259


그부
['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
acc :  0.9259259259259259

['nonflavanoid_phenols', 'total_phenols', 'magnesium', 'hue']
acc :  0.9629629629629629
'''




