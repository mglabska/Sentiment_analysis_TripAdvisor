import pandas as pd
from RNN_models import RNN, Glove, RNNGlove, run_rnn
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn_models import run_sklearn

if __name__ == '__main__':
    data = pd.read_csv('data\preprocessed.csv').dropna()
    X = data["Reviews_cleaned"]
    y = data["Category"]
    algorithms = [
        LogisticRegression(random_state=0),
        KNeighborsClassifier(n_neighbors=20),
        BernoulliNB()
    ]
    networks = [
        RNN(X, y),
        Glove(X, y, data),
        RNNGlove(X, y)
    ]
    run_sklearn(algorithms, X, y)
    run_rnn(networks)
