# Principal Component Analysis (PCA)

# Importing the libraries

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class PrincipalComponentAnalysis:

    def __init__(self, path):  # special method
        self.y = None
        self.X = None
        self.dataset = pd.read_csv(path)
        self.x_test = None
        self.x_train = None
        self.y_test = None
        self.y_train = None
        self.y_pred = None

    def data_preprocessing(self):
        self.X = self.dataset.iloc[:, :-1].values
        self.y = self.dataset.iloc[:, -1].values

        # train_test_split
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.2, random_state=0)

        # feature scaling
        sc = StandardScaler()
        self.x_train = sc.fit_transform(self.x_train)
        self.x_test = sc.transform(self.x_test)

    def apply_pca(self):
        pca = PCA(n_components=2)
        self.x_train = pca.fit_transform(self.x_train)
        self.x_test = pca.transform(self.x_test)
        print(self.x_test)

    def build_model(self):
        classifier = LogisticRegression(random_state=0)
        classifier.fit(self.x_train, self.y_train)
        return classifier

    def evaluate(self, model):
        self.y_pred = model.predict(self.x_test)
        cm = confusion_matrix(self.y_test, self.y_pred)
        print(cm)
        print(accuracy_score(self.y_test, self.y_pred))


path = 'datasets/Wine.csv'
p_object = PrincipalComponentAnalysis(path)
p_object.data_preprocessing()
p_object.apply_pca()
model = p_object.build_model()
p_object.evaluate(model)


