import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# 1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x.shape)  # (150, 4)
print(y.shape)  # (150,)
print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

for j in [MinMaxScaler, StandardScaler]:
    scaler = j()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    print(j.__name__)
    # 2. 모델
    # for i in [LinearSVC, SVC, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier]:
    for i in [LinearSVR(), SVR(), KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor()]:
        print()
        model = i

        # 훈련
        model.fit(x_train,y_train)

        y_pred = model.predict(x_test)
        # print('y_test :', y_test)
        # print('y_pred :', y_pred)

        result = model.score(x_test,y_test)
        print('\'s score(acc) :', result)
        
        # acc = accuracy_score(y_test, y_pred)
        # print(i.__name__ + '\'s accuracy_score :', acc)

        r2 = r2_score(y_test, y_pred)
        print('\'s r2_score :', r2)

    if j == StandardScaler:
        break
    print('=================================================================')

'''
MinMaxScaler

LinearSVC's score(acc) : 0.3031161473087819
LinearSVC's r2_score : -0.01312016432861629

SVC's score(acc) : 0.08498583569405099
SVC's r2_score : 0.09562961909398981

KNeighborsClassifier's score(acc) : 0.17847025495750707
KNeighborsClassifier's r2_score : -0.10407908456457493

DecisionTreeClassifier's score(acc) : 1.0
DecisionTreeClassifier's r2_score : -0.03941797688984994

RandomForestClassifier's score(acc) : 1.0
RandomForestClassifier's r2_score : 0.09762701258986639
=================================================================
StandardScaler

LinearSVC's score(acc) : 0.3937677053824363
LinearSVC's r2_score : 0.2243803780010276

SVC's score(acc) : 0.16147308781869688
SVC's r2_score : 0.04688817022981151

KNeighborsClassifier's score(acc) : 0.17563739376770537
KNeighborsClassifier's r2_score : -0.1395796893608714

DecisionTreeClassifier's score(acc) : 1.0
DecisionTreeClassifier's r2_score : -0.02386606922836365

RandomForestClassifier's score(acc) : 1.0
RandomForestClassifier's r2_score : 0.017031824490906122
'''