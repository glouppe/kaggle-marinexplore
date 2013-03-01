#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin


class MultiFrameClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        self.n_frames = X.shape[1] # expect X to be 3d
        _X, _y = self._flatten(X, y)
        self.base_estimator.fit(_X, _y)

        return self

    def predict(self, X):
        return np.argmax(self.predict_proba(X))

    def predict_proba(self, X):
        _X = self._flatten(X)
        _y = self.base_estimator.predict_proba(_X)

        n_samples = X.shape[0]
        n_classes = _y.shape[1]

        y = np.zeros((n_samples, n_classes))

        for i in xrange(n_samples):
            y[i] = np.sum(_y[i * self.n_frames:(i + 1) * self.n_frames], axis=0) / self.n_frames

        return y

    def _flatten(self, X, y=None):
        shape = X.shape
        _X = X.reshape([shape[0] * shape[1]] + list(shape[2:]))

        if y is None:
            return _X

        else:
            _y = np.hstack(y for i in range(shape[1]))

            return _X, _y


if __name__ == "__main__":
    import numpy as np

    from sklearn.cross_validation import cross_val_score
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.pipeline import Pipeline

    from transform import SpectrogramTransformer
    from transform import WhitenerTransformer

    data = np.load("data/train-subsample.npz")
    X = data["X_train"]
    y = data["y_train"]
    n_samples = len(y)

    pipe = Pipeline([("spectrogram", SpectrogramTransformer(flatten=False, transpose=True, NFFT=1024, noverlap=256, clip=500)),
                     ("whiten", WhitenerTransformer(n_components=None)),
                     ("mfc", MultiFrameClassifier(base_estimator=ExtraTreesClassifier(n_estimators=50)))])

    scores = cross_val_score(pipe, X, y, scoring="roc_auc", cv=3)
    print "%f (+- %f)" % (scores.mean(), scores.std())
