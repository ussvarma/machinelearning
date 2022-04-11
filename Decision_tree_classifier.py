#  Problem Statement:
# Predict whether the cancer is benign or malignant


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
df = pd.read_csv("datasets/cancer.csv")
pd.set_option('display.max_columns', None)

# Exploring dataset
df.head()

df.info()

# dropping id and unnamed column as it does not have any significance
df.drop(["Unnamed: 32", "id"], axis=1, inplace=True)

df.info()

m = df[df["diagnosis"] == "M"]
print(m)

b = df[df.diagnosis == "B"]
print(b)

# visualising the data with their average_radius and texture
plt.title("malignant and benign ")
plt.scatter(m.radius_mean, m.texture_mean, color="red", label="malignant")
plt.scatter(b.radius_mean, b.texture_mean, color="blue", label="malignant")
plt.legend()

# Data preprocessing technique
# label  encoding technique
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["diagnosis"] = le.fit_transform(df["diagnosis"])

# dividing data into independant variable and dependant variable
x = df.drop(["diagnosis"], axis=1)
y = df.diagnosis

# feature scaling
x = (x - np.min(x)) / (np.max(x) - np.min(x))
print(x)

# test train split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# training the model
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {'max_features': ['auto', 'sqrt', 'log2'],
                       'ccp_alpha': [0.1, .01, .001],
                       'max_depth': [5, 6, 7, 8, 9],
                       'criterion': ['gini', 'entropy']
                       }
tree_clas = DecisionTreeClassifier(random_state=1024)
grid_search = RandomizedSearchCV(estimator=tree_clas, param_distributions=param_distributions, cv=5, verbose=True)
grid_search.fit(x, y)

final_model = grid_search.best_estimator_
print(final_model)
y_pred = final_model.predict(x_test)


# performance metric
print("Decision score: ", final_model.score(x_test, y_test))
# Returns the mean accuracy on the given test data and labels.

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

