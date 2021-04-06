import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder,MinMaxScaler,StandardScaler,MaxAbsScaler,PowerTransformer,RobustScaler
from sklearn.model_selection import train_test_split,GridSearchCV,KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

df = pd.read_csv("../data/csv/winequality-white.csv",sep=';')

print(df.iloc[:,-1].value_counts())
print(df.shape)
print(df.describe())

x = df.values[:,:-1]
y = df.values[:,-1]

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=66,shuffle=True, test_size=0.2)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


