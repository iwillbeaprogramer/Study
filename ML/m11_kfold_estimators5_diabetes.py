import warnings
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_diabetes


warnings.filterwarnings('ignore')
datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=44)
allAlgorithms = all_estimators(type_filter = 'regressor')
kfold = KFold(n_splits=5,random_state=77, shuffle=True)
best=[]
for (name,algorithm) in allAlgorithms:
    try:
        model = algorithm()
        scores = cross_val_score(model,x_train, y_train, cv=kfold)        
        print(name, "\'s r2 : \n", scores)
        best.append(max(scores))
    except:
        print(name, '없음')
        continue
print(max(best))
"""
ARDRegression 's r2 : 
 [0.47481641 0.48527985 0.46792353 0.5600166  0.37031528]
AdaBoostRegressor 's r2 : 
 [0.47068877 0.44099738 0.46604176 0.53704431 0.2523477 ]
BaggingRegressor 's r2 : 
 [0.38862726 0.45063329 0.37586964 0.45130044 0.23867103]
BayesianRidge 's r2 :
 [0.49043067 0.47221963 0.47734842 0.56352615 0.37761203]
CCA 's r2 :
 [0.37115606 0.17980581 0.46471171 0.55258411 0.29418323]
DecisionTreeRegressor 's r2 : 
 [ 0.03179616  0.02107664  0.12218619 -0.01584152 -0.3845542 ]
DummyRegressor 's r2 :
 [-0.00432459 -0.01078332 -0.00293931 -0.03545274 -0.00601344]
ElasticNet 's r2 : 
 [ 0.00412805 -0.00069092  0.00541169 -0.02534326  0.00180807]
ElasticNetCV 's r2 : 
 [0.42321988 0.45567821 0.44331144 0.49134338 0.31966629]
ExtraTreeRegressor 's r2 :
 [ 0.22272081  0.04164579 -0.04468065 -0.15805587 -0.47913117]
ExtraTreesRegressor 's r2 : 
 [0.46418919 0.46699936 0.4506921  0.50289136 0.23661705]
GammaRegressor 's r2 :
 [ 0.00162114 -0.00264572  0.00327988 -0.02954757 -0.00031393]
GaussianProcessRegressor 's r2 : 
 [-11.6974413  -27.41969893 -11.043529   -11.65397627 -13.2911592 ]
GeneralizedLinearRegressor 's r2 : 
 [ 0.00179076 -0.00309452  0.00342069 -0.02803447 -0.00051296]
GradientBoostingRegressor 's r2 : 
 [0.5058389  0.4037315  0.45754077 0.5559589  0.25337857]
HistGradientBoostingRegressor 's r2 : 
 [0.41139154 0.41043048 0.40943158 0.47123317 0.23773978]
HuberRegressor 's r2 : 
 [0.49876864 0.45750897 0.45846728 0.54434228 0.40425693]
IsotonicRegression 's r2 :
 [nan nan nan nan nan]
KNeighborsRegressor 's r2 :
 [0.35817382 0.33395457 0.38080499 0.37659325 0.15623429]
KernelRidge 's r2 : 
 [-3.21024896 -3.52623286 -2.96877666 -4.1852857  -3.538776  ]
Lars 's r2 : 
 [ 0.24103098  0.47715173 -2.88338963  0.53499403  0.38655217]
LarsCV 's r2 : 
 [0.45258889 0.47769925 0.47726076 0.55002657 0.370046  ]
Lasso 's r2 :
 [0.34093535 0.35048334 0.32720416 0.36673092 0.30181413]
LassoCV 's r2 : 
 [0.457804   0.47781607 0.47313797 0.56166389 0.36822158]
LassoLars 's r2 :
 [0.38053894 0.39305551 0.36927422 0.42082682 0.32096914]
LassoLarsCV 's r2 : 
 [0.45258889 0.47715146 0.47338409 0.56171182 0.36757517]
LassoLarsIC 's r2 :
 [0.44194233 0.47724265 0.47868004 0.55061817 0.36793503]
LinearRegression 's r2 :
 [0.49652963 0.47715173 0.46565514 0.55184515 0.38655217]
LinearSVR 's r2 :
 [-0.3821754  -0.35273421 -0.41344247 -0.75972889 -0.37262801]
MLPRegressor 's r2 : 
 [-2.60869481 -2.75766432 -2.59821687 -3.49530183 -2.83811122]
MultiOutputRegressor 없음
MultiTaskElasticNet 's r2 :
 [nan nan nan nan nan]
MultiTaskElasticNetCV 's r2 :
 [nan nan nan nan nan]
MultiTaskLasso 's r2 :
 [nan nan nan nan nan]
MultiTaskLassoCV 's r2 : 
 [nan nan nan nan nan]
NuSVR 's r2 :
 [0.1379982  0.14066337 0.12730405 0.10450657 0.10704505]
OrthogonalMatchingPursuit 's r2 : 
 [0.33716192 0.3056527  0.33951272 0.31041136 0.20335767]
OrthogonalMatchingPursuitCV 's r2 : 
 [0.44583958 0.44827293 0.46503415 0.55425066 0.37465953]
PLSCanonical 's r2 : 
 [-0.99728739 -1.99889607 -0.68952862 -1.1601449  -1.86824395]
PLSRegression 's r2 :
 [0.48555824 0.44765396 0.47619168 0.58269817 0.37019117]
PassiveAggressiveRegressor 's r2 : 
 [0.43681275 0.44211281 0.46340092 0.39448112 0.3046228 ]
PoissonRegressor 's r2 :
 [0.30752758 0.35929857 0.34386618 0.37679995 0.2465751 ]
RANSACRegressor 's r2 : 
 [0.28766592 0.02245171 0.11067109 0.13397383 0.29934101]
RadiusNeighborsRegressor 's r2 :
 [-0.00432459 -0.01078332 -0.00293931 -0.03545274 -0.00601344]
RandomForestRegressor 's r2 : 
 [0.4998665  0.41368746 0.46141589 0.53860464 0.28309562]
RegressorChain 없음
Ridge 's r2 :
 [0.38268858 0.42942638 0.40494235 0.4421777  0.29404632]
RidgeCV 's r2 :
 [0.48578296 0.47510598 0.48087375 0.55916314 0.37137981]
SGDRegressor 's r2 : 
 [0.36743712 0.42546448 0.40306031 0.42933134 0.27227608]
SVR 's r2 :
 [0.13146673 0.15492254 0.09948322 0.02698625 0.11497241]
StackingRegressor 없음
TheilSenRegressor 's r2 : 
 [0.48860265 0.45350426 0.46170371 0.54053584 0.40501467]
TransformedTargetRegressor 's r2 :
 [0.49652963 0.47715173 0.46565514 0.55184515 0.38655217]
TweedieRegressor 's r2 : 
 [ 0.00179076 -0.00309452  0.00342069 -0.02803447 -0.00051296]
VotingRegressor 없음
_SigmoidCalibration 's r2 :
 [nan nan nan nan nan]
0.5826981748926718
"""