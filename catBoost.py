# CatBoost model
'''
Documentation of catboost
https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html
'''
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# read the dataset
df = pd.read_csv('datasets/Employee-Attrition.csv')
x = df.drop(['Attrition'], axis=1)
y = df['Attrition']
# train_test_split
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=0)

# print(df)
# find out the indices of categorical variables
categorical_var = np.where(train_x.dtypes == "object")[0]
print('\nCategorical Variables indices : ', categorical_var)

print('\n Training CatBoost Model..........')


model = CatBoostClassifier(iterations=50)

# fit the model with the training data
model.fit(train_x, train_y, cat_features=categorical_var, eval_set=(test_x, test_y), plot=True)

# predict the target on the test dataset
predict_test = model.predict(test_x)

# Accuracy Score on test dataset
accuracy_test = accuracy_score(test_y, predict_test)
print('\naccuracy_score on test dataset : ', accuracy_test)
