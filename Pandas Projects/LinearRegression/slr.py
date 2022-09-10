import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import statsmodels.api as sm

dataset = pd.read_csv('LifeExpectancyData.csv')

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(dataset.iloc[:, 1:])
dataset.iloc[:, 1:] = imputer.transform(dataset.iloc[:, 1:])

dataset = pd.get_dummies(dataset,columns=['Country'],drop_first=True)
#dataset.fillna(-999,inplace=True)

X = dataset.loc[:,dataset.columns.difference(['Life expectancy '],sort=False)].values
y = dataset.iloc[:,19:20].values

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.1,random_state=0)

regrassor = LinearRegression()
regrassor.fit(X_train,y_train)
X_test_pred = regrassor.predict(X_test)

print(regrassor.score(X_test,y_test))

X = np.append(arr=np.ones((X.shape[0],1)).astype(int), values=X,axis=1)

def multiple_regrassor(X,y):
    columns = list(range(X.shape[1]))
    for i in range(X.shape[1]):
        X_opt = np.array(X[:,list(columns)],dtype=float)
        regrassor_ols = sm.OLS(endog=y,exog=X_opt).fit()
        pvalues = list(regrassor_ols.pvalues)
        for j in range(len(pvalues)):
            if(max(pvalues) > 0.05):
                if(pvalues[j] == max(pvalues)):
                    print(j)
                    del(columns[j])
                    break
    return (regrassor_ols,X_opt)
        #print(regressor_ols.summary(),end='\r')

regressor_ols,X_opt = multiple_regrassor(X, y)
print(X_opt.shape)
y4_pred = regressor_ols.predict(X_opt[X_opt.shape[0] - 1,:])

#print(regressor_ols.score(X_opt,y))

'''
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1:].values

dataset.info()
print(dataset.describe())

null_columns = []
for i in range(1,X.shape[0] + 1):
    for j in range(1,X.shape[1] + 1):
        if(str(X[i-1,j-1]) == 'nan' or str(X[i-1,j-1]) == ''):
            #null_columns.append(chr(96 + j))
            null_columns.append(j-1)
    
#print(list(np.sort(np.unique(null_columns))))


imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, [2, 4, 6, 8, 10, 11, 12, 14, 15, 16, 17, 18, 19]])
X[:, [2, 4, 6, 8, 10, 11, 12, 14, 15, 16, 17, 18, 19]] = imputer.transform(X[:, [2, 4, 6, 8, 10, 11, 12, 14, 15, 16, 17, 18, 19]])


labelEncoder = LabelEncoder()
X[:,0] = labelEncoder.fit_transform(X[:,0])


ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X = np.array(ct.fit_transform(X))
X = X[:,1:]
'''