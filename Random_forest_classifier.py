# import dataset from sklearn library
from sklearn.datasets import make_classification

x, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)

# train_test_split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0)
print(x_train.shape, x_test.shape)

from sklearn.ensemble import RandomForestClassifier  # importing randomforestclassifier

rf = RandomForestClassifier(random_state=42, n_jobs=-1)

from sklearn.model_selection import GridSearchCV  # for Hyperparameter tunning


def best_model(params):
    grid_search = GridSearchCV(estimator=rf,
                               param_grid=params,
                               cv=4,
                               n_jobs=-1, verbose=1)

    grid_search.fit(x_train, y_train)
    rf_best = grid_search.best_estimator_
    return rf_best


params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100, 200],
    'n_estimators': [10, 25, 30, 50, 100, 200]
}
y_pred = best_model(params)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
