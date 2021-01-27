from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(1,11)
y = np.array([1,2,4,3,5,5,7,9,8,11])

print(x)
print(y)


optimizer = RMSprop(learning_rate=0.01)
model = Sequential()
model.add(Dense(1,input_shape=(1,)))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse',optimizer=optimizer)
model.fit(x,y,epochs=10)
y_pred = model.predict(x)

plt.scatter(x,y)
plt.plot(x,y_pred,color='red')
plt.show()