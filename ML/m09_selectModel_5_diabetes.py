import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_diabetes


warnings.filterwarnings('ignore')
datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=44)
allAlgorithms = all_estimators(type_filter = 'regressor')
best=[]
for (name,algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        print(name, "\'s accuracy : ", r2_score(y_test,y_pred))
        best.append(r2_score(y_test,y_pred))
    except:
        print(name, '없음')
        continue

print('Best : ',max(best))


'''
Best :  0.5278342233068394
Tensorflow : 0.21290600730061382
'''