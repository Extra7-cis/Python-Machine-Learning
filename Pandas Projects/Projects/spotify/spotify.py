import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



dataset=pd.read_csv('spotify_dataset.csv',low_memory=False,dtype=str)
dataset=dataset.iloc[:,0:4]

print(dataset.isnull().sum())

X=dataset.iloc[0:5000,1:4]
y=dataset.iloc[0:5000,0:1]

X.columns=[0,1,2]

y.columns=[0]

X=X.values
y=y.values



from sklearn.impute import SimpleImputer
simpleimputer=SimpleImputer(missing_values=np.nan, strategy='constant')
simpleimputer.fit(X)
X=simpleimputer.transform(X)


from sklearn.preprocessing import OneHotEncoder
o=OneHotEncoder(sparse=False,handle_unknown='ignore')
o.fit(X)
X1=o.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X1,y,test_size=0.2,random_state=0)



from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)



y_pred=classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)

print('--------------------------------------------------------------')
print('Right :',cm.diagonal().sum())
print('Total :',cm.sum())
print('Accuracy = ',cm.diagonal().sum() / cm.sum())



#accuracy is 92.5%