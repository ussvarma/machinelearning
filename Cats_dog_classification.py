# Problem Statement: Find an optimal ml model for cat-dog image classification

import os
import cv2
# importing libraries
import numpy as np
from tqdm import tqdm

directory = "datasets/PetImages"

categories = ["Dog", "Cat"]

#  Converting image into features


training_data = []
img_size = 100


def create_training_data():
    for category in categories:  # dogs and cats

        path = os.path.join(directory, category)  # create path to dogs and cats
        class_num = categories.index(category)  # classification  (0 or a 1). 0=dog 1=cat

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (img_size, img_size))  # resize to normalize data size
                training_data.append([new_array.flatten(), class_num])  # add this to our training_dat  
            except OSError as e:
                print("OSErrroBad img most likely", e, os.path.join(path, img))
            except Exception as e:
                print("general exception", e, os.path.join(path, img))


create_training_data()

# Shuffling the training data


import random

random.shuffle(training_data)
print(len(training_data))

for sample in training_data[:10]:
    print(sample)

# separating them into independent and dependent variables


sample_1 = training_data[0][0].shape
X = training_data[0][0].reshape(1, -1)

y = [training_data[0][1]]
for features, label in training_data[1:]:
    X = np.concatenate((X, features.reshape(1, -1)), axis=0)
    y.append(label)
print(X)

y = np.array(y)  # converting into array
print(y.shape)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)
print(x_train.shape, x_test.shape, y_test.shape, y_train.shape)

# Normalising the data

x_train = x_train / 255
x_test = x_test / 255

# svc model

from sklearn.svm import SVC

svc = SVC(random_state=0, )
svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)
y_pred = np.reshape(y_pred, (-1, 1))
np.concatenate((y_test.reshape(-1, 1), y_pred.reshape(-1, 1)), axis=1)

from sklearn.metrics import classification_report

print("svm model:",classification_report(y_test, y_pred))

# Logistic Regression model


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=0, solver='saga', n_jobs=-1)
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
y_pred = np.reshape(y_pred, (1495, 1))
np.concatenate((y_test.reshape(-1, 1), y_pred.reshape(-1, 1)), axis=1)

from sklearn.metrics import classification_report

print("logistic regression:",classification_report(y_test, y_pred))

# KNN mdoel


from sklearn.neighbors import KNeighborsClassifier

NN = KNeighborsClassifier()
NN.fit(x_train, y_train)

y_pred = NN.predict(x_test)
y_pred = np.reshape(y_pred, (1495, 1))
np.concatenate((y_test.reshape(-1, 1), y_pred.reshape(-1, 1)), axis=1)

from sklearn.metrics import classification_report

print("Nearest neighbours:",classification_report(y_test, y_pred))

# Naive Bayes model

# In[21]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train, y_train)
y_pred = nb.predict(x_test)

# In[22]:


y_pred = np.reshape(y_pred, (1495, 1))
np.concatenate((y_test.reshape(-1, 1), y_pred.reshape(-1, 1)), axis=1)

# In[23]:


from sklearn.metrics import classification_report

print("Naive Bayes:",classification_report(y_test, y_pred))

#  Decision Tree Model


from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)


y_pred = dtc.predict(x_test)
y_pred = np.reshape(y_pred, (1495, 1))
np.concatenate((y_test.reshape(-1, 1), y_pred.reshape(-1, 1)), axis=1)


from sklearn.metrics import classification_report

print("Desicion tree :",classification_report(y_test, y_pred))

# # Random Forest Model

# In[27]:


from sklearn.ensemble import RandomForestClassifier  # importing randomforestclassifier

rf = RandomForestClassifier(random_state=0, n_jobs=-1)
rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)
y_pred = np.reshape(y_pred, (1495, 1))
np.concatenate((y_test.reshape(-1, 1), y_pred.reshape(-1, 1)), axis=1)


from sklearn.metrics import classification_report

print("Random forest:",classification_report(y_test, y_pred))

# Observation: Both SVC and Random forest model did better than others.
