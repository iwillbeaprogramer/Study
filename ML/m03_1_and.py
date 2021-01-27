from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import accuracy_score
x_data = np.array([[0,0],[0,1],[1,0],[1,1]])
y_data = np.array([0,0,0,1])

model = LinearSVC()

model.fit(x_data,y_data)

y_pred = model.predict(x_data)
result = model.score(x_data,y_data)
print(x_data,"의 예측결과 : ",y_pred)
print('model.score : ',result)
acc = accuracy_score(y_data,y_pred)
print("acc : ",acc)