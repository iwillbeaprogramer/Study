# m31 mnist 0.95 
# ncomponent 1.0

# m31로 만든 0.95 이상의 n_component = ? 를 사용해 dnn모델을 만들것import numpy as np
# mnist dnn보다 성능 좋게 만들어라!!
# cnn과 비교
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import KFold,cross_val_score

params = [
    {"model__n_estimators":[100,200,300],"model__learning_rate":[0.3,0.1,0.01,0.001],"model__max_depth":[4,5,6,],"model__colsample_bytree":[0.6,0.9,1.2],"model__colsample_bylevel":[0.6,0.7,0.8,0.9],}
]

(x_train,y_train),(x_test,y_test) = mnist.load_data()
pca = PCA()
x = np.append(x_train,x_test,axis=0)
x = pca.fit_transform(x.reshape(-1,28*28))/255.0
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum>=0.95)+1



pca = PCA(n_components=d)
x_train = pca.fit_transform(x_train.reshape(-1,784)/255.0)
x_test = pca.transform(x_test.reshape(-1,784)/255.0)
#x_val = pca.transform(x_val.reshape(-1,784)/255.0)

# onehot = OneHotEncoder()
# y_train = onehot.fit_transform(y_train.reshape(-1,1)).toarray()
# y_test = onehot.transform(y_test.reshape(-1,1)).toarray()
# y_val = onehot.transform(y_val.reshape(-1,1)).toarray()
# es = EarlyStopping(monitor = 'val_loss',mode='auto',patience=10)
kfold = KFold(n_splits=5,shuffle=True)


model = XGBClassifier()
search = RandomizedSearchCV(model,params)

print('result : ')
'''
result :  0.9643
'''