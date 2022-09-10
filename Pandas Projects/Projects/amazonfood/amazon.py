import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


##########################################################
#read the dataset
dataset=pd.read_csv('amazonFood.csv')
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
