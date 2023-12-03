__authors__ = ['1638618']
__group__ = 'GM08:30_3'

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from sklearn.model_selection import LeaveOneOut
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


def calculate_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return accuracy, precision, recall, f1


def cross_validation(model, X, y, cv=10):
    scores = cross_val_score(model, X, y, cv=cv)
    return scores


def loocv(model, X, y):
    loo = LeaveOneOut()
    scores = cross_val_score(model, X, y, cv=loo)
    return scores


def main():
    # Load datasets
    dtypesProcessed = {'tweetId': 'int', 'tweetText': 'str', 'tweetDate': 'str', 'sentimentLabel': 'int'}
    dfProcessed = pd.read_csv('FinalStemmedSentimentAnalysisDataset.csv', delimiter=';', dtype=dtypesProcessed)

    print("Comprobar NaNs i limpiarlos eliminando las filas que tienen NaNs")
    NaNsDetector(dfProcessed)
    print("########################### NaN Treatment FINALIZADO ##################################")

    X = dfProcessed['tweetText'].to_numpy()
    y = dfProcessed['sentimentLabel'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("############################## APARTADO C ####################################")
    print("CROSS VALIDATION - METRIC: ACCURACY")
    C_NBT_CV = NaiveBayesText(alpha=1e-10, max_features=None)
    C_NBT_CV.fit(X_train, y_train)
    # Perform cross-validation
    cross_val_scores = cross_validation(C_NBT_CV, X, y, cv=10)
    print("Accuracy Average CV: ", np.mean(cross_val_scores))

    y_pred = C_NBT_CV.predict(X_test)

    # Calculate metrics
    accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)

    # Print the metrics
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    """ LOOCV COMMENTED BECAUSE THE COMPUTATIONAL COST IS TOO BIG.
    print("LOOCV - METRIC: ACCURACY")
    C_NBT_LOOCV = NaiveBayesText(alpha=1e-10, max_features=None)

    # Perform LOOCV
    loocv_scores = loocv(C_NBT_LOOCV, X, y)
    print(np.mean(loocv_scores))
    C_NBT_LOOCV.fit(X_train, y_train)
    y_pred = C_NBT_LOOCV.predict(X_test)

    # Calculate metrics
    accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)

    # Print the metrics
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    """
    print("############################## C FINALIZADO ####################################")

    print("############################## APARTADO B ####################################")
    print("CROSS VALIDATION - METRIC: ACCURACY")
    train_sizes = [0.6, 0.7, 0.8, 0.9]  # Modify this list according to your needs
    dict_sizes = [100000, 500000, 1000000, None]  # Modify this list according to your needs

    for train_size in train_sizes:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_size, random_state=42, stratify=y)
        for dict_size in dict_sizes:
            B_NBT_CV = NaiveBayesText(alpha=1e-10, max_features=dict_size)
            B_NBT_CV.fit(X_train, y_train)
            # Perform cross-validation
            cross_val_scores = cross_validation(B_NBT_CV, X, y, cv=10)
            print(
                f"Train size: {train_size}, Dict size: {dict_size if dict_size is not None else 'all'}, Accuracy Average CV: ",
                np.mean(cross_val_scores))
            y_pred = B_NBT_CV.predict(X_test)

            # Calculate metrics
            accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)

            # Print the metrics
            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")
            print('')
    print("############################## B FINALIZADO ####################################")

    print("############################## APARTADO A ####################################")
    print("CROSS VALIDATION - METRIC: ACCURACY")
    train_sizes = [0.6, 0.7, 0.8, 0.9]  # Modify this list according to your needs
    dict_sizes = [100000, 500000, 1000000, None]  # Modify this list according to your needs

    for train_size in train_sizes:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_size, random_state=42, stratify=y)
        for dict_size in dict_sizes:
            A_NBT_CV = NaiveBayesText(alpha=1, max_features=dict_size)
            A_NBT_CV.fit(X_train, y_train)
            # Perform cross-validation
            cross_val_scores = cross_validation(A_NBT_CV, X, y, cv=10)
            print(
                f"Train size: {train_size}, Dict size: {dict_size if dict_size is not None else 'all'}, Accuracy Average CV: ",
                np.mean(cross_val_scores))
            y_pred = A_NBT_CV.predict(X_test)

            # Calculate metrics
            accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)

            # Print the metrics
            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")
            print('')
    print("############################## A FINALIZADO ####################################")


if __name__ == "__main__":
    main()
