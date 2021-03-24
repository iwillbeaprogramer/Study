import numpy as np
import matplotlib.pyplot as plt

f = lambda x:x**2-4*x+6
x = np.linspace(-1,6,100)
y = f(x)

plt.plot(x,y,'k-')
plt.plot(2,2,'sk')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()