import numpy as np
import pandas as pd

df = pd.read_csv("../data/csv/iris_sklearn.csv",index_col=0,header=0)

print(df)
array = df.values
print(array)

np.save("../data/npy/iris_sklearn.npy",arr=array)

#판다스의 loc, iloc에 대해 정리