import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class NaiveBayesText:
    def __init__(self):
        self.vocab = []
        self.prob_v = {}
        self.prob_w_v = {}

    def fit(self, X, y):
        # Use CountVectorizer to get word counts
        vectorizer = CountVectorizer()
        X_vec = vectorizer.fit_transform(X[:, 1])
        self.vocab = vectorizer.get_feature_names_out()
        V = list(set(y))

        # Calculate the required P(vj) and P(wk | vj) probability terms.
        for vj in V:
            docs_j = [X[i, 1] for i in range(len(X)) if y[i] == vj]
            self.prob_v[vj] = len(docs_j) / len(X)
            text_j = " ".join(docs_j)
            n = len(text_j.split())
            self.prob_w_v[vj] = {}
            for wk in self.vocab:
                nk = text_j.count(wk)
                self.prob_w_v[vj][wk] = (nk + 1) / (n + len(self.vocab))

    def predict(self, Doc):
        positions = [i for i in range(len(self.vocab)) if self.vocab[i] in Doc.split()]
        scores = {}
        for vj in self.prob_v.keys():
            scores[vj] = self.prob_v[vj]
            for i in positions:
                scores[vj] *= self.prob_w_v[vj][self.vocab[i]]
        return max(scores, key=scores.get)
