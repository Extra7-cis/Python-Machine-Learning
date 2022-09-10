
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###### read data
data=pd.read_csv('data.csv')
print(data.isnull().sum())
from sklearn.preprocessing import LabelEncoder
y=LabelEncoder().fit_transform(data.iloc[:,1].values)
x=data.drop(['id','diagnosis'],axis=1).values


#### splite data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_text=train_test_split(x,y,test_size=0.15,random_state=56)

##### StandardScaler to set data in small range
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.fit_transform(x_test)
##############################################
#### choise  LogisticRegression fit and predict
from sklearn.linear_model import LogisticRegression
LR=LogisticRegression(random_state=56)
LR.fit(x_train,y_train)
y_pred=LR.predict(x_test)



###################################################

#### choise  XGBClassifier fit and predict
from xgboost import XGBClassifier
Liner=XGBClassifier(use_label_encoder=False)
Liner.fit(x_train,y_train)
y_pred2=Liner.predict(x_test)


###########################

### choise  RandomForestClassifier fit and predict
from sklearn.ensemble import RandomForestClassifier
RandomForest=RandomForestClassifier(n_estimators=150)
RandomForest.fit(x_train,y_train)
y_pred_RandomForest=RandomForest.predict(x_test)



###################### knn
from sklearn.neighbors import KNeighborsClassifier
KNeighborsClass=KNeighborsClassifier(n_neighbors=6)
KNeighborsClass.fit(x_train,y_train)
y_pred_RKNeighborsClass=KNeighborsClass.predict(x_test)


################# Support Vector Classification
from sklearn.svm import SVC
SVm=SVC()
SVm.fit(x_train,y_train)
y_pred_SVm=SVm.predict(x_test)


import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#####score
print("knn                            : ",accuracy_score(y_text,y_pred_RKNeighborsClass))
print("RandomForestClassifier         : ",accuracy_score(y_text,y_pred_RandomForest))
print("XGBClassifier                  : ",accuracy_score(y_text,y_pred2))
print("LogisticRegression             : ",accuracy_score(y_text,y_pred))
print("Support Vector Classification  : ",accuracy_score(y_text,y_pred_SVm))
#### plot
fig, axes = plt.subplots(2, 3, figsize=(20,20))
sns.heatmap(confusion_matrix(y_text,y_pred2), annot=True,ax=axes[0,0])
axes[0,0].set_title('XgboostClassifier')
sns.heatmap(confusion_matrix(y_text,y_pred_RandomForest), annot=True,ax=axes[0,1])
axes[0,1].set_title('RandomForestClassifier')
sns.heatmap(confusion_matrix(y_text,y_pred), annot=True,ax=axes[1,0])
axes[1,0].set_title('LogisticRegression')
sns.heatmap(confusion_matrix(y_text,y_pred_RKNeighborsClass), annot=True,ax=axes[1,1])
axes[1,1].set_title('KNeighborsRegressor')
sns.heatmap(confusion_matrix(y_text,y_pred_SVm), annot=True,ax=axes[1,2])
axes[1,2].set_title('Support Vector Classification')
plt.show()

######
x=np.insert(x,0,np.ones(569),axis=1)
import statsmodels.api as sm
x_opt=np.array(x[:,[o for o in range(0,31)]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
print(regressor_ols.summary())


