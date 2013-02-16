#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from matplotlib.mlab import specgram

from sklearn.preprocessing import normalize as normalize_
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection


# TODO:
# - MFCC features
# - DBNs features


# Raw features
def raw(X):
    return X

# Normalized raw features
def normalize(X):
    return normalize_(X)

# FFT
def fft(X):
    return abs(np.fft.fft(X))

# FFT real+imag
def fft_realimag(X):
    fft = np.fft.fft(X)
    return np.hstack((fft.real, fft.imag))

# Spectrogram
def spectrogram(X, upper=None):
    _X = []

    for X_i in X:
        s = specgram(X_i, Fs=2000)
        content = s[0]
        freqs = s[1]

        if upper is not None:
            _X.append(content[freqs < upper].flatten())
        else:
            _X.append(content.flatten())

    return np.array(_X)

# Summary statistics of the spectrogram
def spectrogram_stats(X, upper=None):
    _X = []

    for X_i in X:
        s = specgram(X_i, Fs=2000)
        content = s[0]
        freqs = s[1]

        if upper is not None:
            content = content[freqs < upper]

        _X.append(np.hstack((content.min(axis=1),
                             content.max(axis=1),
                             content.mean(axis=1),
                             content.var(axis=1),
                             np.median(content, axis=1))))

    return np.array(_X)

# Gaussian random projection
def gaussian_random_projection(X, eps=0.25):
    return GaussianRandomProjection(eps=eps).fit_transform(X)

# Sparse random projection
def sparse_random_projection(X, eps=0.25):
    return SparseRandomProjection(eps=eps).fit_transform(X)


if __name__ == "__main__":
    from sklearn.cross_validation import cross_val_score
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import SGDClassifier

    # Setup
    data = np.load("data/train-subsample.npz")
    X = data["X_train"]
    y = data["y_train"]

    clf = ExtraTreesClassifier(n_estimators=100, max_features=None, n_jobs=1)
    # clf = SGDClassifier()
    cv = 5

    # Evaluate the goodness of all features
    scores = cross_val_score(clf, raw(X), y, scoring="roc_auc", cv=cv)
    print "X =", scores.mean()

    scores = cross_val_score(clf, normalize(X), y, scoring="roc_auc", cv=cv)
    print "normalize(X) =", scores.mean()

    scores = cross_val_score(clf, fft(X), y, scoring="roc_auc", cv=cv)
    print "fft(X) =", scores.mean()

    scores = cross_val_score(clf, fft(normalize(X)), y, scoring="roc_auc", cv=cv)
    print "fft(normalize(X)) =", scores.mean()

    scores = cross_val_score(clf, fft_realimag(X), y, scoring="roc_auc", cv=cv)
    print "fft_realimag(X) =", scores.mean()

    scores = cross_val_score(clf, spectrogram(X), y, scoring="roc_auc", cv=cv)
    print "spectrogram(X) =", scores.mean()

    for upper in [50, 100, 200, 300, 400, 500, 1000]:
        scores = cross_val_score(clf, spectrogram(X, upper=upper), y, scoring="roc_auc", cv=cv)
        print "spectrogram(X, upper=%d) =" % upper, scores.mean()

    scores = cross_val_score(clf, spectrogram_stats(X), y, scoring="roc_auc", cv=cv)
    print "spectrogram_stats(X) =", scores.mean()

    for upper in [50, 100, 200, 300, 400, 500, 1000]:
        scores = cross_val_score(clf, spectrogram_stats(X, upper=upper), y, scoring="roc_auc", cv=cv)
        print "spectrogram_stats(X, upper=%d) =" % upper, scores.mean()

    scores = cross_val_score(clf, gaussian_random_projection(X), y, scoring="roc_auc", cv=cv)
    print "gaussian_random_projection(X) =", scores.mean()

    scores = cross_val_score(clf, gaussian_random_projection(fft(X)), y, scoring="roc_auc", cv=cv)
    print "gaussian_random_projection(fft(X)) =", scores.mean()

    scores = cross_val_score(clf, sparse_random_projection(X), y, scoring="roc_auc", cv=cv)
    print "sparse_random_projection(X) =", scores.mean()

    scores = cross_val_score(clf, sparse_random_projection(fft(X)), y, scoring="roc_auc", cv=cv)
    print "sparse_random_projection(fft(X)) =", scores.mean()
