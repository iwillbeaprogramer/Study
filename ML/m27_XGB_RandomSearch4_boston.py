import warnings
warnings.filterwarnings('ignore')

params = [
    {"model__n_estimators":[100,200,300],"model__learning_rate":[0.3,0.1,0.01,0.001],"model__max_depth":[4,5,6,],"model__colsample_bytree":[0.6,0.9,1.2],"model__colsample_bylevel":[0.6,0.7,0.8,0.9],}
]
n_jobs=-1
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

datasets = load_boston()
x = datasets.data
y = datasets.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15)

kfold = KFold(n_splits=5,shuffle=True)
model = XGBClassifier(n_jobs = n_jobs)
pipe = Pipeline([('scaler',MinMaxScaler()),('model',model)])
grid = RandomizedSearchCV(pipe,params,cv=kfold)
grid.fit(x_train,y_train)

print(grid.score(x_test,y_test))


