# -*- coding: utf-8 -*-
"""decision_tree_regression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VVh62tu2Hr4wowbZfpoenP44CmgxmbfR

# Decision Tree Regression

## Importing the libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""## Importing the dataset"""

dataset = pd.read_csv('datasets/Position_Salaries.csv')
x = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, -1].values

"""## Training the Decision Tree Regression model on the whole dataset"""

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x, y)

"""## Predicting a new result"""

print(regressor.predict([[5.6], [6], [6.5]]))

"""## Visualising the Decision Tree Regression results (higher resolution)"""

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(-1, 1)
plt.scatter(x, y, color="red")
plt.plot(x_grid, regressor.predict(x_grid), color="blue")
plt.title("Decision Tree Regression")
plt.xlabel("position level")
plt.ylabel("Salary")
plt.show()
