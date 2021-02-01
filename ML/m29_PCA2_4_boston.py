import numpy as np
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA

datasets = load_boston()
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
cumsum>=0.95 :  [0.80582318 0.96887514 0.99022375 0.99718074 0.99848069 0.99920791
 0.99962696 0.9998755  0.99996089 0.9999917  0.99999835 0.99999992
 1.        ]
d :  2
'''