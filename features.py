#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier

data = np.load("data/train-subsample.npz")
X = data["X_train"]
y = data["y_train"]

clf = ExtraTreesClassifier(n_estimators=100, max_features=None, n_jobs=1)
# clf = SGDClassifier()
cv = 5


# Test the goodness of features

# TODO:
# - MFCC features
# - DBNs features

# Raw features
_X = X
_y = y
scores = cross_val_score(clf, _X, _y, scoring="roc_auc", cv=cv)
print "raw =", scores.mean()

# Normalized raw features
from sklearn.preprocessing import normalize
_X = normalize(X)
_y = y
scores = cross_val_score(clf, _X, _y, scoring="roc_auc", cv=cv)
print "normalize(raw) =", scores.mean()

# FFT
_X = abs(np.fft.fft(X))
_y = y
scores = cross_val_score(clf, _X, _y, scoring="roc_auc", cv=cv)
print "fft(raw) =", scores.mean()

# FFT of normalized
_X = abs(np.fft.fft(normalize(X)))
_y = y
scores = cross_val_score(clf, _X, _y, scoring="roc_auc", cv=cv)
print "fft(normalize(raw)) =", scores.mean()

# FFT.real + FFT.imag
fft = np.fft.fft(X)
_X = np.hstack((fft.real, fft.imag))
_y = y
scores = cross_val_score(clf, _X, _y, scoring="roc_auc", cv=cv)
print "fft(raw).real + fft(raw).imag =", scores.mean()

# Spectrogram
from matplotlib.mlab import specgram
spectrogram = []

for X_i in X:
    s = specgram(X_i)
    spectrogram.append(s[0].flatten())

_X = np.array(spectrogram)
_y = y

scores = cross_val_score(clf, _X, _y, scoring="roc_auc", cv=cv)
print "spectrogram =", scores.mean()

# Spectrogram, lower than N hz
from matplotlib.mlab import specgram

for lower_bound in [50, 100, 200, 300, 400, 500, 1000]:
    spectrogram = []

    for X_i in X:
        s = specgram(X_i)
        content = s[0]
        freqs = s[1]
        spectrogram.append(content[freqs <= lower_bound].flatten())

    _X = np.array(spectrogram)
    _y = y

    scores = cross_val_score(clf, _X, _y, scoring="roc_auc", cv=cv)
    print "spectrogram[< %d Hz] =" % lower_bound, scores.mean()

# Random projection
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection

_X = GaussianRandomProjection(eps=0.25).fit_transform(X)
_y = y
scores = cross_val_score(clf, _X, _y, scoring="roc_auc", cv=cv)
print "GaussianRandomProjection(raw) =", scores.mean()

_X = SparseRandomProjection(eps=0.25).fit_transform(X)
_y = y
scores = cross_val_score(clf, _X, _y, scoring="roc_auc", cv=cv)
print "SparseRandomProjection(raw) =", scores.mean()

_X = GaussianRandomProjection(eps=0.25).fit_transform(abs(np.fft.fft(X)))
_y = y
scores = cross_val_score(clf, _X, _y, scoring="roc_auc", cv=cv)
print "GaussianRandomProjection(fft(raw)) =", scores.mean()

_X = SparseRandomProjection(eps=0.25).fit_transform(abs(np.fft.fft(X)))
_y = y
scores = cross_val_score(clf, _X, _y, scoring="roc_auc", cv=cv)
print "SparseRandomProjection(fft(raw)) =", scores.mean()
