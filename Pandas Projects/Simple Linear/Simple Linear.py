#first check data
#check for missing value
#check for strings
#check for feature scaling (wide range in numbers)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('sales.csv')
#X is independent 
X=dataset.iloc[:,:-1].values
#y is dependent (salary) the one we want to predict
y=dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size =0.2,random_state=1) 

#making linear regression 

#regression predicts value in numbers


from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(X_train,y_train)
#predicing through x test
y_pred=regression.predict(X_test)


#draws the points with red color

plt.scatter(X_train,y_train,color='red')
#draws the line in blue 
plt.plot(X_train,regression.predict(X_train),color='blue')
#put title
plt.title('salary vs experiance(training)')
#the x axis title
plt.xlabel('years of experience')
#y axis title
plt.ylabel('salary')
#shows the plot (not neccessary used)
plt.show()
























