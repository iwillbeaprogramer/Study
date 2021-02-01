import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

datasets = load_iris()
x = datasets.data
y = datasets.target
# print(x.shape,y.shape) (442, 10) (442,)

# pca = PCA(n_components=8)
# x2 = pca.fit_transform(x)
# print(x2.shape)

# pca_EVR = pca.explained_variance_ratio_
# print(sum(pca_EVR))

pca = PCA()
pca.fit(x)
cumsum=np.cumsum(pca.explained_variance_ratio_)
print('cumsum : ',cumsum)

d = np.argmax(cumsum>0.95)+1
print('cumsum>=0.95 : ',cumsum)
print("d : ",d)

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()

'''
cumsum :  [0.92461872 0.97768521 0.99478782 1.        ]
cumsum>=0.95 :  [0.92461872 0.97768521 0.99478782 1.        ]
d :  2
'''