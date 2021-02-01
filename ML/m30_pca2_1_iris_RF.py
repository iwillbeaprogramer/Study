import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold,cross_val_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

datasets = load_iris()
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
print("Random_Forest")
print('cumsum : ',cumsum)

d = np.argmax(cumsum>0.95)+1
print('cumsum>=0.95 : ',cumsum)
print("d : ",d)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15)

kfold = KFold(n_splits=5,shuffle=True)

pipe = Pipeline([('scaler',MinMaxScaler()),('model',RandomForestClassifier())])
pipe.fit(x_train,y_train)
print('그냥 scores : ',pipe.score(x_test,y_test))

pipe = Pipeline([('scaler',MinMaxScaler()),('pca',PCA(n_components=d)),('model',RandomForestClassifier())])
pipe.fit(x_train,y_train)
print('PCA scores : ',pipe.score(x_test,y_test))

print("XGBOOST")
d = np.argmax(cumsum>0.95)+1
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15)
kfold = KFold(n_splits=5,shuffle=True)
pipe = Pipeline([('scaler',MinMaxScaler()),('model',XGBClassifier(use_label_encoder=False))])
pipe.fit(x_train,y_train)
print('그냥 scores : ',pipe.score(x_test,y_test))

pipe = Pipeline([('scaler',MinMaxScaler()),('pca',PCA(n_components=d)),('model',XGBClassifier())])
pipe.fit(x_train,y_train)
print('PCA scores : ',pipe.score(x_test,y_test))


'''
RandomForest
cumsum :  [0.92461872 0.97768521 0.99478782 1.        ]
cumsum>=0.95 :  [0.92461872 0.97768521 0.99478782 1.        ]
d :  2
그냥 scores :  0.9565217391304348
pca scores :  0.9565217391304348

XGBOOST
그냥 scores :  0.9565217391304348
PCA scores :  0.9565217391304348
'''