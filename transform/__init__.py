import numpy as np

from matplotlib.mlab import specgram

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA


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
                 log=True):
        self.pad_to = pad_to
        self.NFFT = NFFT
        if noverlap < 1:
            noverlap = int(NFFT * noverlap)
        self.noverlap = noverlap
        self.clip = clip
        self.dtype = dtype
        self.whiten = whiten
        self.log = log

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
                X_prime = np.empty((X.shape[0], Pxx.size), self.dtype)

            X_prime[i, :] = Pxx.flatten()
        return X_prime


## class SpectrogramStatsTransformer(BaseEstimator, TransformerMixin):
##     """Creates summary statistics from the spectrogram representation of X.

##     Arguments
##     ---------

##     """

##     def fit(self, X, y=None, **fit_args):
##         return self

##     def transform(self, X):

##         return X_prime

## # Summary statistics of the spectrogram
## def spectrogram_stats(X, upper=None):
##     _X = []

##     for X_i in X:
##         s = specgram(X_i, Fs=2000)
##         content = s[0]
##         freqs = s[1]

##         if upper is not None:
##             content = content[freqs < upper]

##         _X.append(np.hstack((content.min(axis=1),
##                              content.max(axis=1),
##                              content.mean(axis=1),
##                              content.var(axis=1),
##                              np.median(content, axis=1))))

##     return np.array(_X)
