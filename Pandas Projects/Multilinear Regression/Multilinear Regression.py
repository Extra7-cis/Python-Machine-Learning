"""
dataset x and y
missing value (no missing value)
no feature scaling
split to training and test 
label encoder

"""

import numpy as np
import matplotlib as plt
import pandas as pd


dataset=pd.read_csv('data.csv')
X=dataset.iloc[: ,:-1].values
y=dataset.iloc[: ,4].values



from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])




from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

X=X[:,1:]




## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)

#compare the results in ypred with ytest
#make backward to get rid of unnecessary columns
#add a column of ones
X=np.append(np.ones((50,1)).astype(int),X,axis=1)

#x opt has values that affect model
#fitting model in predictors 
X_opt=X[[0,1,2,3,4,5]]
#backward elimination
#calculate p value
import statsmodels.api as sm
#find p value for each value
X_opt=np.array(X[:,[0,1,2,3,4,5]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()
#we search for biggest p value
#comapre p value with sl (5%)
#if p value is bigger we remove the column
#same but remove 2
X_opt=np.array(X[:,[0,1,3,4,5]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()
#x4 p value is biggest and bigger that SL
X_opt=np.array(X[:,[0,1,3,5]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()
#x1 is biggest and bigger than SL we romove it
X_opt=np.array(X[:,[0,3,5]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size=0.2, random_state=0)
y_predX_opt=regressor_ols.predict(X_test)



"""
import numpy as np
import matplotlib as plt
import pandas as pd


dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[: ,:-1].values
y=dataset.iloc[: ,4].values



from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])




from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

X=X[:,1:]




## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)


X=np.append(np.ones((50,1)).astype(int),X ,axis=1)


import statsmodels.api as sm
X_opt=np.array(X[:,[0,1,2,3,4,5]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()

X_opt=np.array(X[:,[0,1,3,4,5]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()

X_opt=np.array(X[:,[0,3,5]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()



## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train_opt, X_test_opt, y_train_opt, y_test_opt = train_test_split(X_opt, y, test_size=0.2, random_state=0)

y_predX_opt=regressor_ols.predict(X_test_opt)

"""
