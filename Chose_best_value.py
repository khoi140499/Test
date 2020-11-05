from sklearn.model_selection import  GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import  pandas as np
data =np.read_csv("dataset2020 (1).csv")
mapping = {'pe-legit': 0, 'pe-malicious': 1}
data.iloc[:,0].replace(mapping, inplace=True)
X=data.iloc[:,1:].values
Y=data.iloc[:,0].values
X_train ,X_test, Y_train,Y_test = train_test_split(X, Y,test_size=1/5, random_state=0)
n_esimator= [20,30,50,70,100,1000]
max_features=['auto','sqrt','log2']
max_dept=[10,20,30,40,50]
max_dept.append(None)
grid_param={
    'n_estimators':n_esimator,
    'max_features': max_features,
    'max_depth' :max_dept
}
RFR =RandomForestRegressor()
RFR_random=GridSearchCV(estimator=RFR,param_grid=grid_param,cv=5, n_jobs=4)
RFR_random.fit(X_train,Y_train)
print(RFR_random.best_params_)
