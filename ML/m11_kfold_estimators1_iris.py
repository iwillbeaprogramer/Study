import warnings
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_iris


warnings.filterwarnings('ignore')
datasets = load_iris()
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

"""
AdaBoostClassifier 's accuracy : 
 [0.91666667 0.91666667 0.875      1.         0.91666667]
BaggingClassifier 's accuracy : 
 [0.875      0.91666667 0.875      1.         0.95833333]
BernoulliNB 's accuracy :
 [0.16666667 0.29166667 0.29166667 0.25       0.33333333]
CalibratedClassifierCV 's accuracy : 
 [0.95833333 0.83333333 0.75       1.         0.875     ]
CategoricalNB 's accuracy :
 [0.875      0.95833333 0.95833333 0.95833333 0.91666667]
CheckingClassifier 's accuracy :
 [0. 0. 0. 0. 0.]
ClassifierChain 없음
ComplementNB 's accuracy : 
 [0.58333333 0.70833333 0.70833333 0.625      0.66666667]
DecisionTreeClassifier 's accuracy :
 [0.875      0.91666667 0.875      1.         1.        ]
DummyClassifier 's accuracy :
 [0.33333333 0.25       0.58333333 0.16666667 0.20833333]
ExtraTreeClassifier 's accuracy :
 [0.91666667 0.875      0.91666667 0.83333333 0.95833333]
ExtraTreesClassifier 's accuracy : 
 [0.91666667 0.91666667 0.91666667 1.         1.        ]
GaussianNB 's accuracy :
 [1.         0.91666667 0.91666667 0.95833333 1.        ]
GaussianProcessClassifier 's accuracy : 
 [0.91666667 0.95833333 0.95833333 1.         0.95833333]
GradientBoostingClassifier 's accuracy : 
 [0.875      0.91666667 0.875      1.         1.        ]
HistGradientBoostingClassifier 's accuracy : 
 [0.91666667 0.91666667 0.875      1.         1.        ]
KNeighborsClassifier 's accuracy :
 [0.875      0.91666667 0.95833333 1.         1.        ]
LabelPropagation 's accuracy :
 [0.91666667 0.91666667 0.95833333 1.         1.        ]
LabelSpreading 's accuracy : 
 [0.91666667 0.91666667 0.95833333 1.         1.        ]
LinearDiscriminantAnalysis 's accuracy :
 [0.95833333 0.95833333 0.95833333 1.         1.        ]
LinearSVC 's accuracy : 
 [0.95833333 0.91666667 0.875      1.         1.        ]
LogisticRegression 's accuracy : 
 [0.91666667 0.91666667 0.91666667 1.         0.95833333]
LogisticRegressionCV 's accuracy : 
 [0.875      0.91666667 0.91666667 1.         1.        ]
MLPClassifier 's accuracy : 
 [0.91666667 0.95833333 0.95833333 1.         1.        ]
MultiOutputClassifier 없음
MultinomialNB 's accuracy :
 [0.875      0.75       0.75       0.79166667 0.75      ]
NearestCentroid 's accuracy :
 [0.875      0.91666667 0.95833333 0.95833333 0.95833333]
NuSVC 's accuracy :
 [0.91666667 0.95833333 0.91666667 1.         0.91666667]
OneVsOneClassifier 없음
OneVsRestClassifier 없음
OutputCodeClassifier 없음
PassiveAggressiveClassifier 's accuracy : 
 [0.875      0.83333333 0.875      0.625      0.66666667]
Perceptron 's accuracy :
 [0.91666667 0.875      0.875      0.625      0.875     ]
QuadraticDiscriminantAnalysis 's accuracy :
 [0.91666667 0.91666667 0.95833333 1.         0.95833333]
RadiusNeighborsClassifier 's accuracy : 
 [0.95833333 0.91666667 0.91666667 1.         0.95833333]
RandomForestClassifier 's accuracy : 
 [0.875      0.91666667 0.875      1.         1.        ]
RidgeClassifier 's accuracy :
 [0.95833333 0.75       0.79166667 0.875      0.875     ]
RidgeClassifierCV 's accuracy : 
 [0.95833333 0.75       0.79166667 0.875      0.875     ]
SGDClassifier 's accuracy : 
 [0.79166667 0.75       0.91666667 0.66666667 1.        ]
SVC 's accuracy :
 [0.91666667 0.95833333 0.91666667 1.         1.        ]
StackingClassifier 없음
VotingClassifier 없음
"""
