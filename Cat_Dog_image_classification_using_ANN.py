# Problem Statement: cat-dog image classification


# importing libraries
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

pd.set_option('display.max_columns', None)


# Converting image into features


def create_training_data(directory, categories,img_size):
    training_data = []
    for category in categories:  # dogs and cats

        path = os.path.join(directory, category)  # create path to dogs and cats
        class_num = categories.index(category)  # classification  (0 or a 1). 0=dog 1=cat

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path, img))  # convert to array
                new_array = cv2.resize(img_array, (img_size, img_size))  # resize to normalize data size
                training_data.append([new_array.flatten(), class_num])  # add this to our training_dat  
            except OSError as e:
                print("OSErrorBad img most likely", e, os.path.join(path, img))
            except Exception as e:
                print("general exception", e, os.path.join(path, img))
    return training_data


directory = "datasets/PetImages"
categories = ["Dog", "Cat"]
img_size=100
dataset_matrix=create_training_data(directory, categories,img_size)

#  Shuffling the training data

random.shuffle(dataset_matrix)
print(len(dataset_matrix))

for sample in dataset_matrix[:10]:
    print(sample)

# separating them into independent and dependent variables

sample_1 = dataset_matrix[0][0].shape
X = dataset_matrix[0][0].reshape(1, -1)

y = [dataset_matrix[0][1]]
for features, label in dataset_matrix[1:]:
    X = np.concatenate((X, features.reshape(1, -1)), axis=0)
    y.append(label)

print(X)
print(y)

y = np.array(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
print(x_train.shape, x_test.shape, y_test.shape, y_train.shape)

# Normalising the data

x_train = x_train / 255
x_test = x_test / 255


def build_train_model(a, b):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=6, activation='relu'))
    model.add(tf.keras.layers.Dense(units=8, activation='relu'))
    model.add(tf.keras.layers.Dense(units=6, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='Adagrad', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(a, b, batch_size=32, epochs=200)
    return model


ann = build_train_model(x_train, y_train)

# Predicting x_test
y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)
print(y_pred)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# performance metrics
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("accuracy score:", accuracy_score(y_test, y_pred))
