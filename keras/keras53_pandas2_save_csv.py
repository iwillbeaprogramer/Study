import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

dataset = load_iris()
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

x = dataset.data #(150,4)
y = dataset.target #(150,)
df = pd.DataFrame(x,columns = dataset.feature_names)
df.columns = ['sepal_length','sepal_width','petal_length','petal_width']

#y컬럼추가
df['Target'] = y
df.to_csv('../data/csv/iris_sklearn.csv',sep=',')