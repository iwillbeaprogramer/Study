import numpy as np

x = np.array([3,6,5,4,2])

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder()
sklearn_x = onehot.fit_transform(x.reshape(-1,1)).toarray()
keras_x = to_categorical(x)
print(sklearn_x)
print(keras_x)