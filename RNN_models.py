import gensim.downloader
import numpy as np
from gensim.models import KeyedVectors
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import f1_score, cohen_kappa_score, classification_report
from sklearn.model_selection import train_test_split

glove = gensim.downloader.load('glove-twitter-200')
glove.save("glove.model")
glove = KeyedVectors.load("glove.model")
maxlen = 10
max_words = 10000
embedding_dim = 200


class RNN:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.tokenizer = Tokenizer(num_words=max_words)
        self.tokenizer.fit_on_texts(self.X)
        sequences = self.tokenizer.texts_to_sequences(self.X)
        self.X = pad_sequences(sequences, maxlen=maxlen)

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        X_train = pad_sequences(X_train, maxlen=maxlen)
        X_test = pad_sequences(X_test, maxlen=maxlen)
        return X_train, X_test, y_train, y_test

    def fit_model(self):
        X_train, X_test, y_train, y_test = self.split_data()
        model = Sequential()
        model.add(Embedding(max_words, 32))
        model.add(Bidirectional(LSTM(32, return_sequences=True)))
        model.add(LSTM(32))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train,
                  epochs=15,
                  batch_size=128,
                  validation_split=0.2)

        y_pred = np.round(model.predict(X_test))
        print('Wyniki dla RNN z warstwą Embedding:')
        print(classification_report(y_test, y_pred))
        print(f'cohen_kappa_score: {cohen_kappa_score(y_test, y_pred)}')
        print(f'f1_score: {f1_score(y_test, y_pred)}')


class Glove:
    def __init__(self, X, y, data):
        self.X = X
        self.y = y
        self.data = data

    @staticmethod
    def get_vector(sent):
        for word in sent.lower().split(' '):
            if word.isalpha():
                try:
                    return glove[word].tolist()
                except KeyError:
                    return glove['unk'].tolist()
            else:
                return glove['unk'].tolist()


    def pad_trunc(self, vectors, max=maxlen):
        new_data = []
        zero_vector = []
        for _ in range(len(vectors[0][0])):
            zero_vector.append(0.0)
        for sample in vectors:
            if len(sample) > max:
                temp = sample[:max]
            elif len(sample) < max:
                temp = sample
                additional_elems = max - len(sample)
                for _ in range(additional_elems):
                    temp.append(zero_vector)
            else:
                temp = sample
            new_data.append(temp)
        return new_data

    def split_data(self):
        self.data['vectors'] = self.X.apply(lambda x: [self.get_vector(x)])
        self.X = np.array(self.pad_trunc(self.data['vectors'], maxlen))
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=1)
        return X_train, X_test, y_train, y_test

    def fit_model(self):
        X_train, X_test, y_train, y_test = self.split_data()
        model = Sequential()
        model.add(Bidirectional(LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
        model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train,
                  epochs=3,
                  batch_size=16,
                  validation_split=0.2)
        y_pred = np.round(model.predict(X_test))
        print('Wyniki dla RNN z embeddingiem Glove:')
        print(classification_report(y_test, y_pred))
        print(f'cohen_kappa_score: {cohen_kappa_score(y_test, y_pred)}')
        print(f'f1_score: {f1_score(y_test, y_pred)}')


class RNNGlove(RNN):
    def __init__(self, X, y):
        super().__init__(X, y)

    @staticmethod
    def get_index():
        words = list(glove.key_to_index.keys())
        embeddings_index = {}
        for word in words:
            coefs = np.asarray(glove[word], dtype='float32')
            embeddings_index[word] = coefs
        return embeddings_index

    def get_matrix(self):
        embeddings_index = self.get_index()
        word_index = self.tokenizer.word_index
        embedding_matrix = np.zeros((max_words, embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if i < max_words:
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
                else:
                    embedding_matrix[i] = glove['unk']
        return embedding_matrix

    def fit_model(self):
        X_train, X_test, y_train, y_test = self.split_data()
        model = Sequential()
        model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
        model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
        model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.layers[0].set_weights([self.get_matrix()])
        model.layers[0].trainable = False
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.fit(X_train, y_train,
                  epochs=5,
                  batch_size=32,
                  validation_split=0.2)
        y_pred = np.round(model.predict(X_test))
        print('Wyniki dla RNN z warstwą Embedding i embeddingiem Glove:')
        print(classification_report(y_test, y_pred))
        print(f'cohen_kappa_score: {cohen_kappa_score(y_test, y_pred)}')
        print(f'f1_score: {f1_score(y_test, y_pred)}')


def run_rnn(networks):
    for network in networks:
        network.fit_model()
