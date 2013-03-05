#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


def local_maxima(x, step=1):
    if len(x.shape) == 1:
        return ((x >= np.roll(x,  step, 0)) &
                (x >= np.roll(x, -step, 0)))
    else:
        return ((x >= np.roll(x,  step, 0)) &
                (x >= np.roll(x, -step, 0)) &
                (x >= np.roll(x,  step, 1)) &
                (x >= np.roll(x, -step, 1)))

def peaks(x, n_peaks=1, axis=None):
    if axis is None:
        out = np.zeros(n_peaks)
        peaks =  np.sort(x[local_maxima(x)])[::-1]
        n = min(n_peaks, len(peaks))
        out[:n] = peaks[:n]
        return out

    if axis == 0:
        n_bins = x.shape[1]
        x = x.T
    elif axis == 1:
        n_bins = x.shape[0]

    out = np.zeros((n_bins, n_peaks))

    for i in range(n_bins):
        x_i = x[i]
        peaks = np.sort(x_i[local_maxima(x_i)])[::-1]
        n = min(n_peaks, len(peaks))
        out[i, :n] = peaks[:n]

    return out

class PeaksTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, algorithm="max", n_peaks=3, axis=0):
        self.algorithm = algorithm
        self.n_peaks = n_peaks
        self.axis = axis

    def fit(self, X, y=None, **fit_args):
        return self

    def transform(self, X):
        if self.algorithm == "max":
            pass
        elif self.algorithm == "min":
            X = -X

        if self.axis is None:
            n_bins = self.n_peaks
        elif self.axis == 0:
            n_bins = self.n_peaks * X.shape[2]
        elif self.axis == 1:
            n_bins = self.n_peaks * X.shape[1]

        out = np.empty((X.shape[0], n_bins), dtype=np.float32)
        for i, X_i in enumerate(X):
            out[i, :] = peaks(X_i, n_peaks=self.n_peaks, axis=self.axis).flatten()

        if self.algorithm == "max":
            pass
        elif self.algorithm == "min":
            out = -out

        return out


if __name__ == "__main__":
    X = np.random.randint(0, 10, (10, 7, 5))
    print X
    print PeaksTransformer(algorithm="max", n_peaks=3, axis=None).transform(X)
