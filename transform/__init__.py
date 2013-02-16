import numpy as np

from matplotlib.mlab import specgram

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


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
                 clip=1.0, dtype=np.float32):
        self.pad_to = pad_to
        self.NFFT = NFFT
        self.noverlap = noverlap
        self.clip = clip
        self.dtype = dtype

    def fit(self, X, y=None, **fit_args):
        return self

    def transform(self, X):
        X_prime = None
        for i, X_i in enumerate(X):
            s = specgram(X_i, NFFT=self.NFFT, Fs=2, pad_to=self.pad_to,
                         noverlap=self.noverlap)
            Pxx = s[0]
            if self.clip < 1.0:
                freqs = s[1]
                n_fx = freqs.searchsorted(self.clip, side='right')
                Pxx = Pxx[:n_fx]
            if X_prime is None:
                X_prime = np.empty((X.shape[0], Pxx.size), self.dtype)
            X_prime[i, :] = Pxx.flatten()
        return X_prime
