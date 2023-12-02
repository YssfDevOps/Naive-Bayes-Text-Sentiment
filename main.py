__authors__ = ['1638618']
__group__ = 'GM08:30_3'

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from sklearn.naive_bayes import MultinomialNB


from NaiveBayesText import NaiveBayesText


def NaNsDetector(dataset):
    nan_count = dataset.isnull().sum()
    nan_percentage = dataset.isnull().mean() * 100
    print("Número de NaNs por columna:")
    print(nan_count)
    print("\nPorcentaje de NaNs por columna:")
    print(nan_percentage)
    dataset.dropna(inplace=True)


def main():
    # Load datasets
    dtypesProcessed = {'tweetId': 'int', 'tweetText': 'str', 'tweetDate': 'str', 'sentimentLabel': 'int'}
    dfProcessed = pd.read_csv('FinalStemmedSentimentAnalysisDataset.csv', delimiter=';', dtype=dtypesProcessed)

    NaNsDetector(dfProcessed)

    X = dfProcessed['tweetText'].to_numpy()
    y = dfProcessed['sentimentLabel'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    classifier = NaiveBayesText()
    cat_count_arr, prob_table = classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test, cat_count_arr, prob_table)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy SCRATCH: {accuracy}")

    # ------------------------------------------------
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    classifierKK = MultinomialNB()
    classifierKK.fit(X_train_vec, y_train)
    predictionsKK = classifier.predict(X_test_vec)

    # Calculate accuracy
    accuracyr = accuracy_score(y_test, predictionsKK)
    print(f"Accuracy LEGENDARIO: {accuracyr}")


if __name__ == "__main__":
    main()
