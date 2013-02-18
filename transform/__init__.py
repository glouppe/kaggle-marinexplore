import numpy as np

from matplotlib.mlab import specgram

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
    whiten : int or None
        Whether to whiten the spectrogram or not.
        If whiten is not None its an int holding the number
        of components.
    """

    def __init__(self, pad_to=None, NFFT=256, noverlap=200,
                 clip=1000.0, dtype=np.float32, whiten=None,
                 log=True, flatten=True):
        self.pad_to = pad_to
        self.NFFT = NFFT
        if noverlap < 1:
            noverlap = int(NFFT * noverlap)
        self.noverlap = noverlap
        self.clip = clip
        self.dtype = dtype
        self.whiten = whiten
        self.log = log
        self.flatten = flatten

    def fit(self, X, y=None, **fit_args):
        return self

    def transform(self, X):
        X_prime = None
        for i, X_i in enumerate(X):
            s = specgram(X_i, NFFT=self.NFFT, Fs=2000, pad_to=self.pad_to,
                         noverlap=self.noverlap)
            Pxx = s[0]
            if self.log:
                Pxx = 10. * np.log10(Pxx)
            #Pxx = np.flipud(Pxx)
            if self.clip < 1000.0:
                freqs = s[1]
                n_fx = freqs.searchsorted(self.clip, side='right')
                Pxx = Pxx[:n_fx]

            if self.whiten:
                pca = PCA(n_components=self.whiten, whiten=True)
                Pxx = pca.fit_transform(Pxx)

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


class FlattenTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, **fit_args):
        return self

    def transform(self, X):
        out = np.empty((X.shape[0], X.shape[1] * X.shape[2]), dtype=np.float32)
        for i, X_i in enumerate(X):
            out[i, :] = X_i.flatten()
        return out


class SpectrogramStatsTransformer(BaseEstimator, TransformerMixin):
    """Creates summary statistics from the spectrogram representation of X.

    Arguments
    ---------

    """
    def __init__(self):
        def percentile(a, axis=0, p=50):
            return np.percentile(a, p, axis=axis)

        self.stats = [np.min, np.max, np.mean, np.var, np.median,
#                      partial(percentile, p=25), partial(percentile, p=75),
#                      partial(percentile, p=10), partial(percentile, p=90),
                      ]

    def fit(self, X, y=None, **fit_args):
        return self

    def transform(self, X):
        n_stats = len(self.stats)
        n_freqs = X.shape[1]
        out = np.empty((X.shape[0], n_stats * n_freqs), dtype=np.float32)
        for i in xrange(X.shape[0]):
            X_i = X[i]
            for j, stat in enumerate(self.stats):
                vals = stat(X_i, axis=1)
                out[i, n_freqs * j: n_freqs * (j + 1)] = vals
        return out
