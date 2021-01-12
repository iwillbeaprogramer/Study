from sklearn.datasets import load_iris
import numpy as np

datasets = load_iris()
print(datasets.keys())

x_data = datasets.data
#x = datasets['data']
y_data = datasets['target']
# y = datasets.target

print(datasets.frame)
print(datasets.target_names)
print(datasets['filename'])
print(type(x_data),type(y_data))
np.save('./data/iris_x.npy',arr=x_data)
np.save('./data/iris_y.npy',arr=y_data)