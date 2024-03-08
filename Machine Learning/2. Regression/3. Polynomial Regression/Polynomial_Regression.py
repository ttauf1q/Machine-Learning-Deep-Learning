#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X, y)


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polyreg = PolynomialFeatures(degree=4) #finding the X square
x_poly = polyreg.fit_transform(X) #fitting it in x_poly
linreg2 = LinearRegression()  #taking another object linreg2 for our x_poly
linreg2.fit(x_poly, y)  #fitting the object with x_poly


#Visualizing the linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, linreg.predict(X), color = 'blue')
plt.title('Truth or Bluff')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


#Visualizing the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, linreg2.predict(polyreg.fit_transform(X)), color = 'blue')  #Don't use x_poly
plt.title('Truth or Bluff using Polynomial')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()