import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA

datasets = load_diabetes()
x = datasets.data
y = datasets.target
# print(x.shape,y.shape) (442, 10) (442,)

pca = PCA(n_components=8)
x2 = pca.fit_transform(x)
print(x2.shape)

pca_EVR = pca.explained_variance_ratio_
print(sum(pca_EVR))