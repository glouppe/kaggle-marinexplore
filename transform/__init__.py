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
    """

    def __init__(self, pad_to=None, NFFT=256):
        self.pad_to = pad_to
        self.NFFT = NFFT

    def fit(self, X, y=None, **fit_args):
        return self

    def transform(self, X):
        spec_size = 3870
        X_prime = np.empty((X.shape[0], spec_size), dtype=np.float32)
        for i, X_i in enumerate(X):
            s = specgram(X_i, NFFT=self.NFFT, Fs=2, pad_to=self.pad_to)
            X_prime[i, :] = s[0].flatten()
        return X_prime
