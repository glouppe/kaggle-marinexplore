#! /usr/bin/env python
# -*- coding: utf-8 -*-


from sklearn.base import BaseEstimator, ClassifierMixin

from transform import FlattenTransformer


class MultiFrameClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator):
        self.flattener = FlattenTransformer(axis=1)
        self.base_estimator = base_estimator

    def fit(self, X, y):
        self.flattener.fit(X, y)
        self.n_frames = self.flattener.size         # number of frames per example
        _X, _y = self.flattener.transform(X, y)
        self.base_estimator.fit(_X, _y)

        return self

    def predict(self, X):
        return np.argmax(self.predict_proba(X))

    def predict_proba(self, X):
        n_samples = X.shape[0]
        _X = self.flattener(X)
        _y = self.base_estimator.predict_proba(X_)

        y = np.zeros((n_samples, _y.shape[1]))

        for i in xrange(n_samples):
            y[i] = np.sum(_y[i * self.n_frames:(i + 1) * self.n_frames], axis=0) / self.n_frames


if __name__ == "__main__":
    import numpy as np

    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.metrics.scorer import auc_scorer

    from transform import SpectrogramTransformer

    data = np.load("data/train-subsample.npz")
    X = data["X_train"]
    y = data["y_train"]
    n_samples = len(y)

    pipe = Pipeline([("spectrogram", SpectrogramTransformer(flatten=False)),
                     ("mfc", MultiFrameClassifier(base_estimator=ExtraTreesClassifier(n_estimators=10)))])  # tune the forest with mfc__base_estimator__param

    pipe.fit(X, y)
