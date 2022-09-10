#polynomial is when dots are formed in a shape and very far from line
#it is used instead of single linear for better prediction
#we try single linear to see prediction 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("data.csv")
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

# Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
#paste this value of 6.5 level in kernel
lin_reg.predict([[6.5]])
#shows that prediction is not good
# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Single Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#here is the polynomial
from sklearn.preprocessing import PolynomialFeatures
polynomialfeatures=PolynomialFeatures(degree=4)
X_poly=polynomialfeatures.fit_transform(X)


lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

lin_reg2.predict(polynomialfeatures.fit_transform([[6.5]]))

plt.plot(X, lin_reg2.predict(X_poly), color = 'blue')
plt.scatter(X, y, color = 'red')





