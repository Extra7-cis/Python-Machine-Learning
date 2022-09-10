#Breast Cancer Project 

##########################################################
#import for the required libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
##########################################################



##########################################################
#read the dataset
dataset=pd.read_csv('DataSet.csv')
#determine X,y 
#there is 1 dependent variable (diagnosis)
#there are 30 independent variables
X = dataset.iloc[:,2:].values
y = dataset.iloc[:, 1].values 
##########################################################



##########################################################
#Lable encoder for y
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
y=labelencoder.fit_transform(y)
##########################################################



##########################################################
#split the data for train and test 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
##########################################################



##########################################################
#Feature scaling 
from sklearn.preprocessing import StandardScaler
standardscaler=StandardScaler()
X_train=standardscaler.fit_transform(X_train)
X_test=standardscaler.fit_transform(X_test)
##########################################################



##########################################################
#Apply Logestic resgression model 
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
##########################################################
#Confusion matrix to check the accuracy of the model 
from sklearn.metrics import confusion_matrix
cmLR=confusion_matrix(y_test,y_pred)
##########################################################



##########################################################
#KNN METHOD 
from sklearn.neighbors import KNeighborsClassifier
KNNclassifier=KNeighborsClassifier(n_neighbors=5,metric="minkowski",p=2)
KNNclassifier.fit(X_train,y_train)

y_predKNN=KNNclassifier.predict(X_test)

#from sklearn.metrics import confusion_matrix
cmKNN=confusion_matrix(y_test,y_predKNN)
score = classifier.score(X_test,y_test)
print(score)
#########################################################