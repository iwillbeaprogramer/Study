import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold,cross_val_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

datasets = load_diabetes()
x = datasets.data
y = datasets.target
# print(x.shape,y.shape) (442, 10) (442,)

# pca = PCA(n_components=8)
# x2 = pca.fit_transform(x)
# print(x2.shape)

# pca_EVR = pca.explained_variance_ratio_
# print(sum(pca_EVR))

pca = PCA()
pca.fit(x)
cumsum=np.cumsum(pca.explained_variance_ratio_)
print('cumsum : ',cumsum)

d = np.argmax(cumsum>0.95)+1
print('cumsum>=0.95 : ',cumsum)
print("d : ",d)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15)

kfold = KFold(n_splits=5,shuffle=True)

pipe = Pipeline([('scaler',MinMaxScaler()),('model',RandomForestRegressor())])
pipe.fit(x_train,y_train)
print('그냥 scores : ',pipe.score(x_test,y_test))

pipe = Pipeline([('scaler',MinMaxScaler()),('pca',PCA(n_components=d)),('model',RandomForestRegressor())])
pipe.fit(x_train,y_train)
print('PCA scores : ',pipe.score(x_test,y_test))


print('xgboost')
pipe = Pipeline([('scaler',MinMaxScaler()),('model',XGBRegressor())])
pipe.fit(x_train,y_train)
print('그냥 scores : ',pipe.score(x_test,y_test))

pipe = Pipeline([('scaler',MinMaxScaler()),('pca',PCA(n_components=d)),('model',XGBRegressor())])
pipe.fit(x_train,y_train)
print('PCA scores : ',pipe.score(x_test,y_test))

'''
RandomForest
d :  8
그냥 scores :  0.36579199491817826
PCA scores :  0.3754523935093498

xgboost
그냥 scores :  0.37188147811724204
PCA scores :  0.4503550155917574
'''