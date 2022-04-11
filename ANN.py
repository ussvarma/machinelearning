import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=np.inf)


# Data Preprocessing

# Importing the dataset


def data_preprocessing(path):
    dataset = pd.read_csv(path)
    dataset = pd.get_dummies(dataset, columns=["Geography", "Gender"])  # converting categorical variables
    X = dataset.iloc[:, 3:-1].values
    y = dataset.iloc[:, -1].values

    return X, y


path = 'datasets/Churn_Modelling.csv'
X, y = data_preprocessing(path)

print(X)

print(y)


# Splitting the dataset into the Training set and Test set


def data_split(a, b):
    from sklearn.model_selection import train_test_split
    return train_test_split(a, b, test_size=0.2, random_state=0)


X_train, X_test, y_train, y_test = data_split(X, y)

# Feature Scaling


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Building and Training the ANN

def build_train_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=4, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))  # added dropout layer
    model.add(tf.keras.layers.Dense(units=6, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))  # added dropout layer
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=32, epochs=100)
    return model


ann = build_train_model()

# Making the predictions and evaluating the model

print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 0, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

# Predicting the Test set results

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

#  Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("accuracy:score", accuracy_score(y_test, y_pred))

#  Observation:
# 1.Even after adding dropout layer still model is overfitted.
