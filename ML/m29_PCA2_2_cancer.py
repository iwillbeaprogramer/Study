import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA

datasets = load_breast_cancer()
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
cumsum>=0.95 :  [0.98204467 0.99822116 0.99977867 0.9998996  0.99998788 0.99999453
 0.99999854 0.99999936 0.99999971 0.99999989 0.99999996 0.99999998
 0.99999999 0.99999999 1.         1.         1.         1.
 1.         1.         1.         1.         1.         1.
 1.         1.         1.         1.         1.         1.        ]
d :  1
'''