#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from matplotlib import mlab
from scipy.stats import skew
from scipy import signal

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA

import IPython


class SpectrogramTransformer(BaseEstimator, TransformerMixin):
    """Creates a flattened spectrogram representation of X.

    Arguments
    ---------
    pad_to : int or None
        The number of points to which the data segment is padded when
        performing the FFT. If None same as ``NFFT``.
    NFFT : int
        The number of data points used in each block for the FFT.
    noverlap : int
        overlap of sliding windows - must be smaller than NFFT.
        The higher the smoother but the more comp intensive.
    clip : float
        Clip frequencies higher than ``clip``.
    dtype : np.dtype
        The dtype of the resulting array.
    """

    def __init__(self, pad_to=None, NFFT=256, noverlap=200,
                 clip=1000.0, dtype=np.float32,
                 log=True, flatten=True, transpose=False,
                 window=None):
        self.pad_to = pad_to
        self.NFFT = NFFT
        if noverlap < 1:
            noverlap = int(NFFT * noverlap)
        self.noverlap = noverlap
        self.clip = clip
        self.dtype = dtype
        self.log = log
        self.flatten = flatten
        self.transpose = transpose
        self.window = window

    def fit(self, X, y=None, **fit_args):
        return self

    def transform(self, X):
        X_prime = None

        window = self.window
        if window is None:
            window = mlab.window_hanning
        if window == 'none':
            window = mlab.window_none

        for i, X_i in enumerate(X):
            Pxx, freqs, _ = mlab.specgram(X_i, NFFT=self.NFFT, Fs=2000,
                                          pad_to=self.pad_to,
                                          noverlap=self.noverlap,
                                          window=window)

            if self.log:
                Pxx = 10. * np.log10(Pxx)

            if self.clip < 1000.0:
                n_fx = freqs.searchsorted(self.clip, side='right')
                Pxx = Pxx[:n_fx]

            if self.transpose:
                Pxx = Pxx.T

            if X_prime is None:
                if self.flatten:
                    X_prime = np.empty((X.shape[0], Pxx.size), self.dtype)
                else:
                    X_prime = np.empty((X.shape[0], Pxx.shape[0],
                                        Pxx.shape[1]), self.dtype)

            if self.flatten:
                Pxx = Pxx.flatten()
                X_prime[i, :] = Pxx
            else:
                X_prime[i, :, :] = Pxx

        return X_prime


class StatsTransformer(BaseEstimator, TransformerMixin):
    """Creates summary statistics from X."""

    def __init__(self, axis=1):
        def percentile(a, axis=0, p=50):
            return np.percentile(a, p, axis=axis)

        self.stats = [np.min, np.max, np.mean, np.var, np.median, np.ptp]
        #self.stats = [np.max]
        self.axis = axis

    def fit(self, X, y=None, **fit_args):
        return self

    def transform(self, X):
        n_stats = len(self.stats)
        if self.axis == 0:
            n_bins = X.shape[2]
        elif self.axis == 1:
            n_bins = X.shape[1]
        out = np.empty((X.shape[0], n_stats * n_bins), dtype=np.float32)
        for i in xrange(X.shape[0]):
            X_i = X[i]
            for j, stat in enumerate(self.stats):
                vals = stat(X_i, axis=self.axis)
                out[i, n_bins * j: n_bins * (j + 1)] = vals
        return out


class WhitenerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y=None, **fit_args):
        _X, _y = self._flatten(X, y)
        self.pca = PCA(n_components=self.n_components, whiten=True)
        self.pca.fit(_X, _y)

        return self

    def transform(self, X):
        _X = self.pca.transform(self._flatten(X))

        return _X.reshape(list(X.shape[:-1]) + [-1])

    def _flatten(self, X, y=None, axis=1):
        shape = X.shape
        _X = X.reshape([shape[0] * shape[1]] + list(shape[2:]))

        if y is None:
            return _X

        else:
            _y = np.hstack(y for i in range(shape[1]))
            return _X, _y


class FlattenTransformer(BaseEstimator, TransformerMixin):
    """Flattens X from 3d to 2d."""

    def __init__(self, scale=1.0):
        self.scale = scale

    def fit(self, X, y=None, **fit_args):
        return self

    def transform(self, X):
        out = np.empty((X.shape[0], X.shape[1] * X.shape[2]), dtype=np.float32)
        for i, X_i in enumerate(X):
            out[i, :] = X_i.flatten()

        out *= self.scale
        return out


class FuncTransformer(BaseEstimator, TransformerMixin):
    """Flattens X from 3d to 2d, apply func, and reshape to 3d."""

    def __init__(self, func, **func_args):
        self.func = func
        self.func_args = func_args

    def fit(self, X, y=None, **fit_args):
        return self

    def transform(self, X):
        _X = X.reshape((X.shape[0], -1))
        _Xt = self.func(X, **self.func_args)
        _X = _Xt.reshape(X.shape)

        return _X


class FilterTransformer(BaseEstimator, TransformerMixin):
    """Applies a filter function to each row in ``X``

    Example
    -------

    >>> tf = FilterTransformer(scipy.signal.wiener)
    >>> X = tf.fit_transform(X)
    """

    def __init__(self, filter_, *filter_args, **kwargs):
        self.filter = filter_
        self.filter_args = filter_args
        self.noise = kwargs.pop('noise', False)

    def fit(self, X, y=None, **fit_args):
        return self

    def transform(self, X):
        out = np.empty_like(X)
        filter_ = self.filter
        filter_args = self.filter_args
        noise = None
        if self.noise:
            noise = 1e-4 * np.random.rand(X.shape[1]) - 0.5
        for i, x in enumerate(X):
            if noise is not None:
                x += noise
            out[i] = filter_(x, *filter_args)
        return out


class DiffTransformer(BaseEstimator, TransformerMixin):
    """Difference features (aka delta's). """

    def __init__(self, w=9):
        self.w = w

    def fit(self, X, y=None, **fit_args):
        return self

    def transform(self, X):
        hlen = int(np.floor(self.w / 2.0))
        self.w = 2 * hlen + 1
        win = np.arange(hlen, -(hlen + 1), -1)
        out = np.zeros_like(X)

        for i in xrange(X.shape[0]):
            x = X[i]
            # pad signal
            A = np.tile(x[0, :], (hlen, 1))
            C = np.tile(x[-1, :], (hlen, 1))
            xx = np.r_[A, x, C]

            # apply filter
            d = signal.lfilter(win, 1, xx, axis=1)

            # trim padding
            d = d[hlen:-hlen]

            out[i] = d

        return out


class LogTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, **fit_args):
        return self

    def transform(self, X):
        return np.log10(X)


class TemplateMatcher(BaseEstimator, TransformerMixin):

    def __init__(self, raw=True):
        self.raw = raw

    def fit(self, X, y=None, **fit_args):
        X_pos = X[y == 1]

        # median pos image
        self.template = [
            np.mean(X_pos, axis=0)[10:30, 5:20],
            np.mean(X_pos, axis=0)[5:35, 5:20],
            np.mean(X_pos, axis=0)[10:30, 2:25],
            np.mean(X_pos, axis=0)[5:35, 2:25],
                         ]
        return self

    def transform(self, X):
        from skimage.feature import match_template
        X_out = None
        n_templates = len(self.template)
        raw = self.raw
        for i, x in enumerate(X):
            if i % 1000 == 0:
                print i

            for j, template in enumerate(self.template):
                result = match_template(x, template, pad_input=True)
                if X_out is None:
                    if raw:
                        dtype = (X.shape[0], n_templates, result.shape[0],
                                 result.shape[1])
                    else:
                        dtype = (X.shape[0], n_templates)
                    X_out = np.empty(dtype, dtype=np.float32)

                if not raw:
                    result = np.max(result)

                X_out[i, j] = result

        if raw:
            X_out = np.max(X_out, axis=1)
        return X_out
