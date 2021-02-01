import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split,KFold,GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

params = [
    {"model__n_estimators":[100,200,300],"model__learning_rate":[0.3,0.1,0.01,0.001],"model__max_depth":[4,5,6,],"model__colsample_bytree":[0.6,0.9,1.2],"model__colsample_bylevel":[0.6,0.7,0.8,0.9],}
]

datasets = load_boston()
x = datasets.data
y = datasets.target
kfold = KFold(n_splits=5,shuffle=True)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15)
pca = PCA()
pca.fit(x)
cumsum=np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum>0.99)+1


pipeline = Pipeline([('scaler',MinMaxScaler()),('pca',PCA(n_components=d)),('model',XGBRegressor())])
grid = GridSearchCV(pipeline,params,cv=kfold)

grid.fit(x_train,y_train)
print(grid.score(x_test,y_test))

