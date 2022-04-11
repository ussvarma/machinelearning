
# Problem Statement:
# Predict whether the cancer is benign or malignant


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
df = pd.read_csv("datasets/cancer.csv")
pd.set_option('display.max_columns', None)

# Exploring dataset
print(df.head())

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
plt.show()

# Data preprocessing technique
# label  encoding technique
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["diagnosis"] = le.fit_transform(df["diagnosis"])

# dividing data into independent variable and dependant variable
x = df.drop(["diagnosis"], axis=1)
y = df.diagnosis
print(x)

print(y)

# feature scaling
x = (x - np.min(x)) / (np.max(x) - np.min(x))
print(x)

# test train split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# training the model
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train, y_train)
y_pred = nb.predict(x_test)

# performance metric
print("Naive Bayes score: ", nb.score(x_test, y_test))
# Returns the mean accuracy on the given test data and labels.


from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))

# In[21]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
