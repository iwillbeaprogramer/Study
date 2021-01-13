import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

dataset = load_iris()
print(dataset.keys())
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
print(dataset.values())
print(dataset.target_names)

x = dataset.data #(150,4)
y = dataset.target #(150,)
df = pd.DataFrame(x,columns = dataset.feature_names)
print(df)
print(df.columns)
print(df.index) 
print(df.head()) # df[:5]
print(df.tail())
print(df.info())
print(df.describe())

df.columns = ['sepal_length','sepal_width','petal_length','petal_width']
print(df.columns)

#y컬럼추가
df['Target'] = dataset.target
print(df.head())

print(df.head())
print(df.columns)
print(df.index)
print(df.tail())
print(df.isnull)
print(df.isnull().sum())


# 상관계수 히트맵
print(df.corr())
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(font_scale=1.2)
# sns.heatmap(data=df.corr(),square=True,annot=True,cbar=True)
# plt.show()

# 도수 분포도
#print(df['Target'].value_counts())
plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
plt.hist(x='sepal_length',data = df)
plt.title('sepal_length')

plt.subplot(2,2,2)
plt.hist(x='sepal_width',data=df)
plt.title('sepal_width')

plt.subplot(2,2,3)
plt.hist(x='petal_length',data=df)
plt.title('petal_length')

plt.subplot(2,2,4)
plt.hist(x='petal_width',data=df)
plt.title('petal_width')
plt.show()