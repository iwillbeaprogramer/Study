import warnings
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_boston


warnings.filterwarnings('ignore')
datasets = load_boston()
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
 [0.74662694 0.63754332 0.66286116 0.77005862 0.61576365]
AdaBoostRegressor 's r2 : 
 [0.86986224 0.8049647  0.8427118  0.80468451 0.85094505]
BaggingRegressor 's r2 : 
 [0.84466211 0.83946438 0.85682215 0.86801432 0.83211444]
BayesianRidge 's r2 :
 [0.71658716 0.62985707 0.71696677 0.78396776 0.62350229]
CCA 's r2 :
 [0.75403154 0.60370745 0.59151525 0.78236827 0.58652477]
DecisionTreeRegressor 's r2 : 
 [0.7707204  0.82340855 0.62270428 0.78271447 0.70506746]
DummyRegressor 's r2 :
 [-1.55205849e-02 -2.43226047e-06 -2.59834446e-02 -9.13739266e-04
 -2.34589093e-04]
ElasticNet 's r2 :
 [0.6663248  0.60657827 0.69831825 0.72523377 0.61139593]
ElasticNetCV 's r2 : 
 [0.65514074 0.60044384 0.69084761 0.70803604 0.60162794]
ExtraTreeRegressor 's r2 :
 [0.77664328 0.65527174 0.51249537 0.77804681 0.74944095]
ExtraTreesRegressor 's r2 : 
 [0.89638348 0.8562542  0.8924921  0.90974786 0.79880865]
GammaRegressor 's r2 :
 [-1.65136734e-02 -2.49157998e-06 -2.37543752e-02 -9.81300631e-04
 -2.55467699e-04]
GaussianProcessRegressor 's r2 : 
 [-5.08066529 -6.33484961 -8.92429454 -5.37834544 -5.86897332]
GeneralizedLinearRegressor 's r2 : 
 [0.64742523 0.58256814 0.68820964 0.67985025 0.61415328]
GradientBoostingRegressor 's r2 : 
 [0.90033174 0.8457656  0.89821205 0.86366919 0.87397139]
HistGradientBoostingRegressor 's r2 : 
 [0.88454538 0.75772677 0.87713592 0.86276842 0.82856316]
HuberRegressor 's r2 : 
 [0.65069225 0.5971038  0.62180112 0.71994347 0.55609058]
IsotonicRegression 's r2 :
 [nan nan nan nan nan]
KNeighborsRegressor 's r2 :
 [0.43657373 0.46585783 0.49681309 0.51124179 0.35861471]
KernelRidge 's r2 : 
 [0.70299474 0.5972021  0.65493216 0.77100787 0.59802237]
Lars 's r2 : 
 [0.74785581 0.66040768 0.67547128 0.77676921 0.62684136]
LarsCV 's r2 : 
 [0.71739278 0.6572376  0.68701485 0.77674681 0.6065536 ]
Lasso 's r2 : 
 [0.64800235 0.60573242 0.69177829 0.69882376 0.60581517]
LassoCV 's r2 : 
 [0.68015752 0.61539379 0.70518945 0.74152654 0.61933109]
LassoLars 's r2 :
 [-1.55205849e-02 -2.43226047e-06 -2.59834446e-02 -9.13739266e-04
 -2.34589093e-04]
LassoLarsCV 's r2 : 
 [0.74566122 0.66040768 0.70295399 0.77671252 0.62863883]
LassoLarsIC 's r2 :
 [0.74393361 0.65931048 0.68016808 0.77672388 0.61619103]
LinearRegression 's r2 : 
 [0.74785581 0.66040768 0.70254877 0.77676921 0.6270103 ]
LinearSVR 's r2 : 
 [0.3833879  0.50752082 0.09629049 0.30578619 0.53183271]
MLPRegressor 's r2 : 
 [0.42318417 0.58034768 0.6101552  0.6333713  0.1849688 ]
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
 [0.15139792 0.17046999 0.33906496 0.24070346 0.14202207]
OrthogonalMatchingPursuit 's r2 :
 [0.55854059 0.48106068 0.53678613 0.54362952 0.50300862]
OrthogonalMatchingPursuitCV 's r2 : 
 [0.69254896 0.55930871 0.61750308 0.78339249 0.58740262]
PLSCanonical 's r2 : 
 [-1.23408589 -2.66867359 -4.51618046 -1.30502955 -2.53460814]
PLSRegression 's r2 :
 [0.69178866 0.6239101  0.64923557 0.79299719 0.56480161]
PassiveAggressiveRegressor 's r2 : 
 [ 0.00091165  0.36786225  0.18709555  0.13707846 -0.26553909]
PoissonRegressor 's r2 : 
 [0.76621304 0.67335206 0.76122483 0.83308468 0.70325536]
RANSACRegressor 's r2 : 
 [0.67808604 0.48503229 0.47801438 0.80665426 0.47285374]
RadiusNeighborsRegressor 없음
RandomForestRegressor 's r2 : 
 [0.88770166 0.82142731 0.86121855 0.86457768 0.86415202]
RegressorChain 없음
Ridge 's r2 :
 [0.73471493 0.65022198 0.71567619 0.78341214 0.6254167 ]
RidgeCV 's r2 : 
 [0.74573605 0.65901402 0.70585334 0.77845332 0.62731105]
SGDRegressor 's r2 :
 [-9.05516098e+24 -1.31965941e+26 -2.26466477e+26 -4.78472931e+26
 -6.77791018e+26]
SVR 's r2 : 
 [0.11585384 0.13322653 0.32796875 0.1983922  0.11640294]
StackingRegressor 없음
TheilSenRegressor 's r2 : 
 [0.68870906 0.58373363 0.65129342 0.79209163 0.5596807 ]
TransformedTargetRegressor 's r2 :
 [0.74785581 0.66040768 0.70254877 0.77676921 0.6270103 ]
TweedieRegressor 's r2 : 
 [0.64742523 0.58256814 0.68820964 0.67985025 0.61415328]
VotingRegressor 없음
_SigmoidCalibration 's r2 :
 [nan nan nan nan nan]
0.9097478634023458
"""