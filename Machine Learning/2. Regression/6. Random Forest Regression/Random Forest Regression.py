# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators = 300, random_state= 0)
reg.fit(X, y)


# Visualizing the Random Forest Regression results
X_grid = np.arange(min(X),max(X), 0.1 )
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, reg.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff using Random Forest Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

