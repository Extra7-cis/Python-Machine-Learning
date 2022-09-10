#python and dataset should be in same file
#only py and dataset are required

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('data1.csv')
#choosing all rows and all columns except the lat one (-1 can be replaced by 2)
#iloc to send columns numbers 
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
#configures missing data
from sklearn.impute import SimpleImputer
#replacing missing data with mean
simpleimputer=SimpleImputer(missing_values=np.nan,strategy="mean")
#setting the missing data in second and third column
simpleimputer.fit(X[:,1:3])
X[:,1:3]=simpleimputer.transform(X[:,1:3])
#importing label encoder so that it turns strings into numbers
#making label encoder to first row and column 0
#we do not put x in variable because we want to split it to dummy variables
from sklearn.preprocessing import LabelEncoder
#making object from class to use
labelEncoder_X=LabelEncoder()
#use fit to set data to all rows in the first column (countries)
labelEncoder_X.fit_transform(X[:,0])
#making label encoder for y (purchased)
#making yet and no 1s and 0s
labelEncoder_y=LabelEncoder()
#we set y to a variable because we don't need to make dummy
y=labelEncoder_y.fit_transform(y)
print(y)

#column transformer slices the columns 
#hot encoder puts the columns 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct= ColumnTransformerct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#chooses random variable for testing 
#setting random to a value in order to be same test as dr
from sklearn.model_selection import train_test_split
#creating train and test for each X and y
#putting values of X and y and (text siz)e for test and random 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size =0.2,random_state=1) 


#feature scalin making all data the same scale
#for age in big range 
#salaries in big range 
#big difference between them
#putting the data in the same scale
#setting age and salary columns
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train[:,3:]=sc.fit_transform(X_train[:,3:])
X_test[:,3:]=sc.transform(X_test[:,3:])













































"""

# -*- coding: utf-8 -*-

Created on Thu Jun 10 20:38:42 2021

@author:Ruba Alkhusheiny

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.impute import SimpleImputer
simpleimputer=SimpleImputer( missing_values=np.nan,strategy="mean")
simpleimputer.fit(X[:,1:3])
X[:,1:3]=simpleimputer.transform(X[:,1:3])


from sklearn.preprocessing import LabelEncoder
labelEncoder_X= LabelEncoder()
labelEncoder_X.fit_transform(X[:,0])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct= ColumnTransformerct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))






from sklearn.preprocessing import LabelEncoder
labelEncoder_y= LabelEncoder()
y = labelEncoder_y.fit_transform(y)
print(y)
"""






