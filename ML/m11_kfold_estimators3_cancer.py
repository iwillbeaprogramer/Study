import warnings
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_breast_cancer


warnings.filterwarnings('ignore')
datasets = load_breast_cancer()
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
 [0.97802198 0.95604396 0.95604396 0.91208791 0.95604396]
BaggingClassifier 's accuracy : 
 [0.97802198 0.93406593 0.97802198 0.92307692 0.96703297]
BernoulliNB 's accuracy :
 [0.61538462 0.6043956  0.58241758 0.61538462 0.68131868]
CalibratedClassifierCV 's accuracy : 
 [0.92307692 0.93406593 0.89010989 0.9010989  0.94505495]
CategoricalNB 없음
CheckingClassifier 's accuracy :
 [0. 0. 0. 0. 0.]
ClassifierChain 없음
ComplementNB 's accuracy :
 [0.87912088 0.9010989  0.84615385 0.86813187 0.91208791]
DecisionTreeClassifier 's accuracy : 
 [0.87912088 0.94505495 0.96703297 0.85714286 0.91208791]
DummyClassifier 's accuracy :
 [0.46153846 0.58241758 0.50549451 0.45054945 0.57142857]
ExtraTreeClassifier 's accuracy : 
 [0.87912088 0.96703297 0.95604396 0.84615385 0.93406593]
ExtraTreesClassifier 's accuracy : 
 [0.98901099 0.96703297 0.97802198 0.94505495 0.93406593]
GaussianNB 's accuracy :
 [0.93406593 0.89010989 0.97802198 0.91208791 0.95604396]
GaussianProcessClassifier 's accuracy : 
 [0.86813187 0.93406593 0.91208791 0.87912088 0.9010989 ]
GradientBoostingClassifier 's accuracy : 
 [0.98901099 0.94505495 0.96703297 0.87912088 0.94505495]
HistGradientBoostingClassifier 's accuracy : 
 [0.98901099 0.95604396 0.98901099 0.92307692 0.95604396]
KNeighborsClassifier 's accuracy :
 [0.89010989 0.95604396 0.91208791 0.89010989 0.92307692]
LabelPropagation 's accuracy : 
 [0.3956044  0.41758242 0.42857143 0.3956044  0.32967033]
LabelSpreading 's accuracy : 
 [0.3956044  0.41758242 0.42857143 0.3956044  0.32967033]
LinearDiscriminantAnalysis 's accuracy :
 [0.94505495 0.96703297 0.95604396 0.94505495 0.94505495]
LinearSVC 's accuracy : 
 [0.86813187 0.93406593 0.83516484 0.92307692 0.92307692]
LogisticRegression 's accuracy : 
 [0.9010989  0.96703297 0.92307692 0.93406593 0.93406593]
LogisticRegressionCV 's accuracy : 
 [0.96703297 0.96703297 0.97802198 0.95604396 0.94505495]
MLPClassifier 's accuracy : 
 [0.94505495 0.93406593 0.85714286 0.92307692 0.93406593]
MultiOutputClassifier 없음
MultinomialNB 's accuracy :
 [0.87912088 0.9010989  0.84615385 0.85714286 0.91208791]
NearestCentroid 's accuracy :
 [0.86813187 0.91208791 0.89010989 0.84615385 0.91208791]
NuSVC 's accuracy : 
 [0.84615385 0.87912088 0.89010989 0.84615385 0.9010989 ]
OneVsOneClassifier 없음
OneVsRestClassifier 없음
OutputCodeClassifier 없음
PassiveAggressiveClassifier 's accuracy : 
 [0.87912088 0.89010989 0.86813187 0.89010989 0.92307692]
Perceptron 's accuracy :
 [0.75824176 0.9010989  0.67032967 0.9010989  0.89010989]
QuadraticDiscriminantAnalysis 's accuracy : 
 [0.96703297 0.95604396 0.97802198 0.91208791 0.93406593]
RadiusNeighborsClassifier 없음
RandomForestClassifier 's accuracy : 
 [0.98901099 0.95604396 0.96703297 0.92307692 0.95604396]
RidgeClassifier 's accuracy :
 [0.92307692 0.93406593 0.95604396 0.92307692 0.95604396]
RidgeClassifierCV 's accuracy : 
 [0.94505495 0.93406593 0.96703297 0.94505495 0.96703297]
SGDClassifier 's accuracy : 
 [0.85714286 0.91208791 0.85714286 0.9010989  0.94505495]
SVC 's accuracy : 
 [0.93406593 0.94505495 0.9010989  0.84615385 0.92307692]
StackingClassifier 없음
VotingClassifier 없음
'''