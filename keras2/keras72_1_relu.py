import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0,x)

x = np.arange(-5,5,0.1)
y = relu(x)

plt.plot(x,y,)
plt.grid()
plt.show()

## 과제
## elu,selu,leaky_relu
## 72-2,3,4로 숙제