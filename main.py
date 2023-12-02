__authors__ = ['1638618']
__group__ = 'GM08:30_3'

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import sys
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import numpy as np
from NaiveBayesText import NaiveBayesText


def NaNsDetector(dataset):
    nan_count = dataset.isnull().sum()
    nan_percentage = dataset.isnull().mean() * 100
    print("Número de NaNs por columna:")
    print(nan_count)
    print("\nPorcentaje de NaNs por columna:")
    print(nan_percentage)
    nan_rows = dataset[dataset.isnull().any(axis=1)]
    print("\nFilas con NaNs:")
    print(nan_rows)
    dataset.dropna(inplace=True)


def main():
    # Load datasets
    dtypesProcessed = {'tweetId': 'int', 'tweetText': 'str', 'tweetDate': 'str', 'sentimentLabel': 'int'}
    dfProcessed = pd.read_csv('FinalStemmedSentimentAnalysisDataset.csv', delimiter=';', dtype=dtypesProcessed)
    """
    dtypesOriginal = {'ItemID': 'int', 'Sentiment': 'int', 'SentimentSource': 'str', 'SentimentText': 'str'}
    dfOriginal = pd.read_csv('SentimentAnalysisDataset.csv', delimiter=',', dtype=dtypesOriginal)
    """

    X = dfProcessed.drop('sentimentLabel', axis=1).to_numpy()
    y = dfProcessed['sentimentLabel'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train[:, 1] = X_train[:, 1].astype(str)
    X_test[:, 1] = X_test[:, 1].astype(str)

    clf = NaiveBayesText()
    clf.fit(X_train, y_train)

    # Predict the sentiment labels for the test data
    y_pred = clf.predict(X_test)


if __name__ == "__main__":
    main()
