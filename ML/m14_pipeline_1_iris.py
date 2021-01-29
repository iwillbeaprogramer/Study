from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


datasets = load_iris()
x = datasets.data
y = datasets.target


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15)
for i in [MinMaxScaler(),StandardScaler()]:
    pipeline = Pipeline([('scaler',i),('model',RandomForestClassifier())])
    pipeline.fit(x_train,y_train)
    print('pipeline score : ',pipeline.score(x_test,y_test))

'''
pipeline score :  0.9565217391304348
pipeline score :  0.9565217391304348
'''