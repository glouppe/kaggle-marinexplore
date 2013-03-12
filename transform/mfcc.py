import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

import yaafelib as yf


class MFCCTransformer(BaseEstimator, TransformerMixin):
    """Extract MFCC features using yaafelib"""

    def __init__(self, sample_rate=16000, diff=False, diff2=False):
        self.diff = diff
        self.diff2 = diff2
        self.sample_rate = sample_rate

    def fit(self, X, y=None, **fit_args):

        return self

    def transform(self, X):
        X_prime = None

        params = {'block_size': 512,
                  'step_size': 256,
                  'mel_min_freq': 130.0,
                  'ceps_ign_first_coef': 0,
                  'fft_len': 0,
                  }

        fp = yf.FeaturePlan(sample_rate=self.sample_rate)
        fp.addFeature('mfcc: MFCC blockSize=%(block_size)d stepSize=%(step_size)d'
                      ' MelMinFreq=%(mel_min_freq)f '
                      ' CepsIgnoreFirstCoeff=%(ceps_ign_first_coef)d' %
                      params)
        if self.diff:
            fp.addFeature('mfcc_d1: MFCC blockSize=%(block_size)d'
                          ' stepSize=%(step_size)d'
                          '> Derivate DOrder=1' % params)
        if self.diff2:
            fp.addFeature('mfcc_d2: MFCC blockSize=%(block_size)d'
                          ' stepSize=%(step_size)d'
                          '> Derivate DOrder=2' % params)

        fp.addFeature('melspec: MelSpectrum FFTWindow=Hanning  MelNbFilters=40'
                      ' blockSize=%(block_size)d stepSize=%(step_size)d'
                      ' MelMinFreq=%(mel_min_freq)f' % params)

        ## fp.addFeature('energy: Energy blockSize=%(block_size)d'
        ##               ' stepSize=%(step_size)d' % params)

        ## fp.addFeature('specs: SpectralSlope FFTLength=%(fft_len)d  FFTWindow=Hanning'
        ##               ' blockSize=%(block_size)d stepSize=%(step_size)d' %
        ##               params)

        ## fp.addFeature('sss: SpectralShapeStatistics FFTLength=0  FFTWindow=Hanning  '
        ##               ' blockSize=%(block_size)d stepSize=%(step_size)d' %
        ##               params)

        ## fp.addFeature('tss: TemporalShapeStatistics '
        ##               ' blockSize=%(block_size)d stepSize=%(step_size)d' %
        ##               params)

        ## fp.addFeature('lpc: LPC LPCNbCoeffs=2 blockSize=%(block_size)d stepSize=%(step_size)d' % params)

        df = fp.getDataFlow()
        engine = yf.Engine()
        engine.load(df)

        X = X.astype(np.float64)
        x_shape = (1, X.shape[1])

        for i, x in enumerate(X):
            x = x.reshape(x_shape)

            feats = engine.processAudio(x)

            if X_prime is None:
                fx_groups = tuple(feats.keys())
                n_features = 0
                for fx_group in fx_groups:
                    n_features += feats[fx_group].ravel().shape[0]

                X_prime = np.empty((X.shape[0], n_features), dtype=np.float64)
                print 'n_groups:', len(fx_groups)
                print 'n_features:', n_features

            offset = 0
            for fx_group in fx_groups:
                fxs = feats[fx_group].ravel()
                X_prime[i, offset:(offset + fxs.shape[0])] = fxs
                offset += fxs.shape[0]

        return X_prime
