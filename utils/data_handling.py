import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score

def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

class PipelineLabelBinarizer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.encoder = LabelBinarizer()
        self.encoder.fit(X)
        self.classes_ = self.encoder.classes_
        return self
    def transform(self, X, y = 0):
        return self.encoder.transform(X)

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

class FillNaWith(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_name, replacement='Null'):
        self.replacement = replacement
        self.attribute_name = attribute_name
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_name].fillna(self.replacement)

class CabinTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = LabelBinarizer();

    def fit(self, X, y=None):
        self.encoder.fit(self.get_cabin_chars(X))
        return self

    def transform(self, X, y=None):
        cabins_tr = []
        # expecting cabin is only column
        cabin_strings = X[:, 0]
        for cabin_string in cabin_strings:
            cabins = cabin_string.split(" ")
            n_cabins = len(cabins)

            try:
                cabin_num = int(cabins[0][1:])
            except:
                cabin_num = -1

            cabins_tr.append([cabin_num, n_cabins])

        cabin_chars = self.get_cabin_chars(X)
        cabin_char_1hot = self.encoder.transform(cabin_chars)

        return np.c_[cabin_char_1hot, cabins_tr]

    def get_cabin_chars(self, X):
        cabin_chars = []
        # expecting cabin is only column
        cabin_strings = X[:, 0]
        for cabin_string in cabin_strings:
            cabins = cabin_string.split(" ")
            try:
                cabin_char = cabins[0][0]
            except IndexError:
                cabin_char = "X"
            cabin_chars.append(cabin_char)

        return cabin_chars

    def get_classes(self):
        return list(self.encoder.classes_) + ["cabin_number", "n_cabins"]

def unwrap_labels(input):
    result = []
    for row in input:
        result.append(row[0])
    return result

def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())

def predict(estimator, test_set, true_labels):
    predictions = estimator.predict(test_set)
    prediction_probabilities = estimator.predict_proba(test_set)

    correct = true_labels - predictions == 0
    n_correct = np.sum(correct)
    print("Correct: ", n_correct)
    print("%Correct: ", 100 * n_correct / len(predictions))

    auc = roc_auc_score(true_labels, prediction_probabilities[:, 1])
    print("ROC AUC: ", auc)

    return auc