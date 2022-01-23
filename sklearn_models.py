from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, cohen_kappa_score, classification_report
from sklearn.model_selection import train_test_split


class SklearnModel:
    def __init__(self, algorithm, X, y):
        self.algorithm = algorithm
        self.X = X
        self.y = y

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        vectorizer = TfidfVectorizer(lowercase=False)
        vectorizer.fit(X_train)
        X_train = vectorizer.transform(X_train)
        X_test = vectorizer.transform(X_test)
        return X_train, X_test, y_train, y_test

    def fit_model(self):
        X_train, X_test, y_train, y_test = self.split_data()
        model = self.algorithm.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f'Wyniki dla {self.algorithm.__class__.__name__}:')
        print(classification_report(y_test, y_pred))
        print(f'cohen_kappa_score: {cohen_kappa_score(y_test, y_pred)}')
        print(f'f1_score: {f1_score(y_test, y_pred)}')


def run_sklearn(algorithms, X, y):
    for algorithm in algorithms:
        model = SklearnModel(algorithm, X, y)
        model.fit_model()
