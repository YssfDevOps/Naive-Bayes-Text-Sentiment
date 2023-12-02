import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer


class NaiveBayesText():
    def __init__(self, alpha=1, max_features=None):
        self.alpha = alpha  # Laplace Smoothing
        self.Cnt_Vec = CountVectorizer(max_features=max_features)
        self.lbl = LabelBinarizer()

    def fit(self, X_train, y_train):
        BOW_train = self.Cnt_Vec.fit_transform(X_train)
        train_Y = self.lbl.fit_transform(y_train)
        if train_Y.shape[1] == 1:
            train_Y = np.concatenate([1 - train_Y, train_Y], axis=1)

        # Calculate cat_count_arr without storing it as an instance variable
        cat_count_arr = np.log(np.sum(train_Y, axis=0) / np.sum(train_Y))

        # Use sparse matrix multiplication directly
        consolidated_train_df = train_Y.T @ BOW_train

        prob_table_numer = consolidated_train_df + self.alpha
        prob_table_denom = np.sum(prob_table_numer, axis=1)
        prob_table = np.log(prob_table_numer) - np.log(prob_table_denom.reshape(-1, 1))

        # Return cat_count_arr and prob_table
        return cat_count_arr, prob_table

    def predict(self, X_test, cat_count_arr, prob_table):
        BOW_test = self.Cnt_Vec.transform(X_test)
        predict_arr = self.lbl.classes_[np.argmax(BOW_test @ prob_table.T + cat_count_arr, axis=1)]
        return predict_arr