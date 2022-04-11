# -*- coding: utf-8 -*-
"""Support_vector_regression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15eGfhRLQvr7N4oCC7dbsBc7PEV3qS59W

# Support Vector Regression (SVR)

## Importing the libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""## Importing the dataset"""

dataset = pd.read_csv("datasets/Position_Salaries.csv")
x = dataset.iloc[:, :1].values
y = dataset.iloc[:, -1].values

print(x)

print(y)

y = y.reshape(-1, 1)
print(y)  # converted y feature into 2dimensional

"""## Feature Scaling"""

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

print(x)

print(y)

"""## Training the SVR model on the whole dataset"""

from sklearn.svm import SVR

regressor = SVR(kernel="rbf")
regressor.fit(x, y.ravel())

"""## Predicting a new result"""

sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1, 1))
# as predict object gives 1d array , we are converting it into 2d array

"""## Visualising the SVR results"""

plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color="red")
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x).reshape(-1, 1)), color='blue')
plt.title("SVR")
plt.xlabel("Position")
plt.ylabel("salary")
plt.show()

"""## Visualising the SVR results (for higher resolution and smoother curve)"""

x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape(-1, 1)  # converting into 2D
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color="red")
plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid)).reshape(-1, 1)), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

"""# Disadvantage of svr model:
It cannot handle outliers

compulsory feature scaling should be done
"""
