import warnings
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_wine


warnings.filterwarnings('ignore')
datasets = load_wine()
x = datasets.data
y = datasets.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=44)
allAlgorithms = all_estimators(type_filter = 'classifier')
kfold = KFold(n_splits=5,random_state=77, shuffle=True)
best=[]
for (name,algorithm) in allAlgorithms:
    try:
        model = algorithm()
        scores = cross_val_score(model,x_train, y_train, cv=kfold)        
        print(name, "\'s accuracy : \n", scores)
        best.append(max(scores))
    except:
        print(name, '없음')
        continue
print(max(best))


'''
AdaBoostClassifier 's accuracy : 
 [0.82758621 0.96551724 0.96428571 0.75       0.39285714]
BaggingClassifier 's accuracy : 
 [0.96551724 0.96551724 0.96428571 0.92857143 0.92857143]
BernoulliNB 's accuracy :
 [0.37931034 0.4137931  0.32142857 0.42857143 0.42857143]
CalibratedClassifierCV 's accuracy : 
 [0.93103448 0.89655172 0.96428571 0.89285714 0.89285714]
CategoricalNB 없음
CheckingClassifier 's accuracy : 
 [0. 0. 0. 0. 0.]
ClassifierChain 없음
ComplementNB 's accuracy :
 [0.5862069  0.65517241 0.57142857 0.71428571 0.75      ]
DecisionTreeClassifier 's accuracy : 
 [0.89655172 0.96551724 0.92857143 0.85714286 0.89285714]
DummyClassifier 's accuracy :
 [0.34482759 0.44827586 0.35714286 0.17857143 0.5       ]
ExtraTreeClassifier 's accuracy :
 [0.93103448 0.96551724 0.85714286 0.96428571 0.92857143]
ExtraTreesClassifier 's accuracy : 
 [1.         1.         1.         0.96428571 0.96428571]
GaussianNB 's accuracy :
 [1.         1.         1.         0.92857143 1.        ]
GaussianProcessClassifier 's accuracy : 
 [0.51724138 0.48275862 0.46428571 0.5        0.32142857]
GradientBoostingClassifier 's accuracy : 
 [0.89655172 1.         0.96428571 0.89285714 0.92857143]
HistGradientBoostingClassifier 's accuracy : 
 [0.96551724 1.         1.         0.96428571 1.        ]
KNeighborsClassifier 's accuracy : 
 [0.68965517 0.62068966 0.71428571 0.75       0.67857143]
LabelPropagation 's accuracy :
 [0.31034483 0.44827586 0.42857143 0.35714286 0.46428571]
LabelSpreading 's accuracy :
 [0.31034483 0.44827586 0.42857143 0.35714286 0.46428571]
LinearDiscriminantAnalysis 's accuracy :
 [1.         0.96551724 1.         0.96428571 0.96428571]
LinearSVC 's accuracy :
 [0.82758621 0.75862069 0.67857143 0.75       0.60714286]
LogisticRegression 's accuracy :
 [0.89655172 1.         1.         0.89285714 0.92857143]
LogisticRegressionCV 's accuracy :
 [0.93103448 1.         1.         0.92857143 0.92857143]
MLPClassifier 's accuracy :
 [0.89655172 0.51724138 1.         0.53571429 0.89285714]
MultiOutputClassifier 없음
MultinomialNB 's accuracy :
 [0.82758621 0.79310345 0.82142857 0.92857143 0.82142857]
NearestCentroid 's accuracy :
 [0.68965517 0.68965517 0.64285714 0.78571429 0.89285714]
NuSVC 's accuracy :
 [0.96551724 0.79310345 0.78571429 0.96428571 0.92857143]
OneVsOneClassifier 없음
OneVsRestClassifier 없음
OutputCodeClassifier 없음
PassiveAggressiveClassifier 's accuracy :
 [0.75862069 0.34482759 0.32142857 0.57142857 0.25      ]
Perceptron 's accuracy :
 [0.5862069  0.68965517 0.53571429 0.46428571 0.71428571]
QuadraticDiscriminantAnalysis 's accuracy : 
 [1.         1.         0.89285714 1.         1.        ]
RadiusNeighborsClassifier 없음
RandomForestClassifier 's accuracy : 
 [1.         1.         1.         0.96428571 1.        ]
RidgeClassifier 's accuracy :
 [1.         1.         1.         0.96428571 1.        ]
RidgeClassifierCV 's accuracy : 
 [1.         1.         1.         0.96428571 1.        ]
SGDClassifier 's accuracy : 
 [0.72413793 0.65517241 0.46428571 0.39285714 0.53571429]
SVC 's accuracy : 
 [0.75862069 0.65517241 0.67857143 0.75       0.85714286]
StackingClassifier 없음
VotingClassifier 없음
'''