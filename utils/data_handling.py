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