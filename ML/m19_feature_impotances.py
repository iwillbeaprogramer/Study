from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 데이터
datasets = load_iris()
x_train,x_test,y_train,y_test = train_test_split(datasets.data,datasets.target,test_size=0.15)

# 모델
model = DecisionTreeClassifier(max_depth = 4)

# 훈련
model.fit(x_train,y_train)

# 평가, 예측
acc = model.score(x_test,y_test)
print(model.feature_importances_)
print("acc : ",acc)
