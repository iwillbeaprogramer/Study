from sklearn.datasets import load_boston,load_diabetes,load_breast_cancer,load_iris,load_wine
from tensorflow.keras.datasets import mnist,fashion_mnist,cifar10,cifar100
import numpy as np

# 1.boston
boston_datasets = load_boston()
boston_x = boston_datasets.data
boston_y = boston_datasets.target
np.save('./data/boston_x.npy',arr=boston_x)
np.save('./data/boston_y.npy',arr=boston_y)
# 2.diabetes
diabetes_datasets = load_diabetes()
diabetes_x = diabetes_datasets.data
diabetes_y = diabetes_datasets.target
np.save('./data/diabetes_x.npy',arr=diabetes_x)
np.save('./data/diabetes_y.npy',arr=diabetes_y)
# 3.cancer
cancer_datasets = load_breast_cancer()
cancer_x = cancer_datasets.data
cancer_y = cancer_datasets.target
np.save('./data/cancer_x.npy',arr=cancer_x)
np.save('./data/cancer_y.npy',arr=cancer_y)
# 4.iris
iris_datasets = load_iris()
iris_x = iris_datasets.data
iris_y = iris_datasets.target
np.save('./data/cancer_x.npy',arr=cancer_x)
np.save('./data/cancer_y.npy',arr=cancer_y)
# 5.wine
wine_datasets = load_wine()
wine_x = wine_datasets.data
wine_y = wine_datasets.target
np.save('./data/wine_x.npy',arr=wine_x)
np.save('./data/wine_y.npy',arr=wine_y)
# 6.mnist
(mnist_x_train,mnist_y_train),(mnist_x_test,mnist_y_test) = mnist.load_data()
np.save('./data/mnist_x_train.npy',arr=mnist_x_train)
np.save('./data/mnist_x_test.npy',arr=mnist_x_test)
np.save('./data/mnist_y_train.npy',arr=mnist_y_train)
np.save('./data/mnist_y_test.npy',arr=mnist_y_test)
# 7.fashion_mnist
(fmnist_x_train,fmnist_y_train),(fmnist_x_test,fmnist_y_test) = fashion_mnist.load_data()
np.save('./data/fmnist_x_train.npy',arr=fmnist_x_train)
np.save('./data/fmnist_x_test.npy',arr=fmnist_x_test)
np.save('./data/fmnist_y_train.npy',arr=fmnist_y_train)
np.save('./data/fmnist_y_test.npy',arr=fmnist_y_test)
# 8.cifar10
(cifar10_x_train,cifar10_y_train),(cifar10_x_test,cifar10_y_test) = cifar10.load_data()
np.save('./data/cifar10_x_train.npy',arr=cifar10_x_train)
np.save('./data/cifar10_x_test.npy',arr=cifar10_x_test)
np.save('./data/cifar10_y_train.npy',arr=cifar10_y_train)
np.save('./data/cifar10_y_test.npy',arr=cifar10_y_test)
# 9.cifar100
(cifar100_x_train,cifar100_y_train),(cifar100_x_test,cifar100_y_test) = cifar100.load_data()
np.save('./data/cifar100_x_train.npy',arr=cifar100_x_train)
np.save('./data/cifar100_x_test.npy',arr=cifar100_x_test)
np.save('./data/cifar100_y_train.npy',arr=cifar100_y_train)
np.save('./data/cifar100_y_test.npy',arr=cifar100_y_test)