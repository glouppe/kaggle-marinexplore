#! /usr/bin/env python
# -*- coding: utf-8 -*-


from sklearn.base import BaseEstimator, ClassifierMixin


def _flatten(X, y=None, axis=1):
    shape = X.shape
    _X = X.reshape([shape[0] * shape[1]] + list(shape[2:]))

    if y is None:
        return _X

    else:
        _y = np.hstack(y for i in range(shape[1]))

        return _X, _y


class MultiFrameClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, transformer=None):
        self.base_estimator = base_estimator
        self.transformer = transformer

    def fit(self, X, y):
        self.n_frames = X.shape[1] # expect X to be 3d
        _X, _y = _flatten(X, y)

        if self.transformer is not None:
            self.transformer.fit(_X, _y)
            _X = self.transformer.transform(_X)

        self.base_estimator.fit(_X, _y)

        return self

    def predict(self, X):
        return np.argmax(self.predict_proba(X))

    def predict_proba(self, X):
        _X = _flatten(X)

        if self.transformer is not None:
            _X = self.transformer.transform(_X)

        _y = self.base_estimator.predict_proba(_X)

        n_samples = X.shape[0]
        n_classes = _y.shape[1]

        y = np.zeros((n_samples, n_classes))

        for i in xrange(n_samples):
            y[i] = np.sum(_y[i * self.n_frames:(i + 1) * self.n_frames], axis=0) / self.n_frames

        return y


if __name__ == "__main__":
    import numpy as np

    from sklearn.decomposition import PCA
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.cross_validation import cross_val_score

    from transform import SpectrogramTransformer

    data = np.load("data/train-subsample.npz")
    X = data["X_train"]
    y = data["y_train"]
    n_samples = len(y)

    pipe = Pipeline([("spectrogram", SpectrogramTransformer(flatten=False, transpose=True)),
                     ("mfc", MultiFrameClassifier(base_estimator=ExtraTreesClassifier(n_estimators=10), # tune the forest with mfc__base_estimator__param
                                                  transformer=None))])                                  # tune PCA with mfc__transformer__param

    scores = cross_val_score(pipe, X, y, scoring="roc_auc", cv=3)
    print "%f (+- %f)" % (scores.mean(), scores.std())
