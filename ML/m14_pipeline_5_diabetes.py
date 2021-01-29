from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


datasets = load_diabetes()
x = datasets.data
y = datasets.target


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15)
for i in [MinMaxScaler(),StandardScaler()]:
    pipeline = Pipeline([('scaler',i),('model',RandomForestRegressor())])
    pipeline.fit(x_train,y_train)
    print('pipeline score : ',pipeline.score(x_test,y_test))

'''
pipeline score :  0.4536120034045238
pipeline score :  0.42258635670629796
'''