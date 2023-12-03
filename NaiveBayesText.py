import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer


class NaiveBayesText():
    def __init__(self, alpha=1, max_features=None):
        self.max_features = max_features
        self.alpha = alpha  # Laplace Smoothing
        self.vectorizer = CountVectorizer(max_features=self.max_features)
        self.label_binarizer = LabelBinarizer()
        self.log_prior = None
        self.log_likelihood = None

    def get_params(self, deep=True):
        return {"alpha": self.alpha, "max_features": self.max_features}

    def score(self, X, y_test):
        y_pred = self.predict(X)
        return accuracy_score(y_test, y_pred)

    def fit(self, X_train, y_train):
        # Transform the text data into a dictionary
        bag_of_words_train = self.vectorizer.fit_transform(X_train)

        # Transform the labels into a binary format
        binarized_labels = self.label_binarizer.fit_transform(y_train)

        # If there are only two classes, duplicate the column to have two columns
        if binarized_labels.shape[1] == 1:
            binarized_labels = np.concatenate([1 - binarized_labels, binarized_labels], axis=1)

        # Calculate the prior for each class
        self.log_prior = np.log(np.sum(binarized_labels, axis=0) / np.sum(binarized_labels))

        # Calculate the frequency for each class
        term_frequency_per_class = binarized_labels.T @ bag_of_words_train

        # Calculate the likelihood
        likelihood_numerator = term_frequency_per_class + self.alpha
        likelihood_denominator = np.sum(likelihood_numerator, axis=1)
        self.log_likelihood = np.log(likelihood_numerator) - np.log(likelihood_denominator.reshape(-1, 1))

    def predict(self, X_test):
        # Transform the test data into a bag-of-words (like a dictionary)
        bag_of_words_test = self.vectorizer.transform(X_test)

        # Calculate the posterior probability for each class and return the class with the highest probability
        predicted_class = self.label_binarizer.classes_[
            np.argmax(bag_of_words_test @ self.log_likelihood.T + self.log_prior, axis=1)]
        return predicted_class
