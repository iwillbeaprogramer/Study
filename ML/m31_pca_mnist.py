import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

(x_train,_),(x_test,_) = mnist.load_data()
x = np.append(x_train,x_test,axis=0)

# 실습
# pca를 통해 0.95 이상인거 몇개?
# pca  배운거 다 집어넣어서 확인
pca = PCA()

x = pca.fit_transform(x.reshape(-1,28*28))
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum>=0.99)+1
print(d)
plt.plot(cumsum)
plt.grid()
plt.show()
#pca.plot_importance
#plt.plot()


