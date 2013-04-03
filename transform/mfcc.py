import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

import yaafelib as yf


class MFCCTransformer(BaseEstimator, TransformerMixin):
    """Extract MFCC features using yaafelib"""

    def __init__(self, sample_rate=16000, diff=False, diff2=False, slope=False):
        self.diff = diff
        self.diff2 = diff2
        self.sample_rate = sample_rate
        self.slope = slope

    def fit(self, X, y=None, **fit_args):

        return self

    def transform(self, X):
        X_prime = None

        params = {'block_size': 256,
                  'step_size': 128,
                  'mel_min_freq': 0.0,
                  'mel_max_freq': 500.0,
                  'mel_nb_filters': 50,
                  'ceps_ign_first_coef': 0,
                  'fft_len': 0,
                  'do1len': 5,
                  'do2len': 1,
                  'slope_step_nbframes': 5,
                  'slope_nbframes': 9,
                  }

        fp = yf.FeaturePlan(sample_rate=self.sample_rate)

        fp.addFeature('melspec: MelSpectrum FFTWindow=Hanning  MelNbFilters=%(mel_nb_filters)d'
                      ' blockSize=%(block_size)d stepSize=%(step_size)d'
                      ' MelMinFreq=%(mel_min_freq)f MelMaxFreq=%(mel_max_freq)f' % params)

        if self.diff:
            fp.addFeature('melspec_diff1: MelSpectrum FFTWindow=Hanning  MelNbFilters=%(mel_nb_filters)d'
                      ' blockSize=%(block_size)d stepSize=%(step_size)d'
                      ' MelMinFreq=%(mel_min_freq)f MelMaxFreq=%(mel_max_freq)f'
                      ' > Derivate DOrder=1 DO1Len=%(do1len)d' % params)

        if self.diff2:
            fp.addFeature('melspec_diff2: MelSpectrum FFTWindow=Hanning  MelNbFilters=%(mel_nb_filters)d'
                      ' blockSize=%(block_size)d stepSize=%(step_size)d'
                      ' MelMinFreq=%(mel_min_freq)f MelMaxFreq=%(mel_max_freq)f'
                      ' > Derivate DOrder=2 DO2Len=%(do2len)d' % params)

        if self.slope:
            fp.addFeature('melspec_slope: MelSpectrum FFTWindow=Hanning  MelNbFilters=%(mel_nb_filters)d'
                      ' blockSize=%(block_size)d stepSize=%(step_size)d'
                      ' MelMinFreq=%(mel_min_freq)f MelMaxFreq=%(mel_max_freq)f'
                      ' > SlopeIntegrator NbFrames=%(slope_nbframes)d  StepNbFrames=%(slope_step_nbframes)d' % params)

        df = fp.getDataFlow()
        engine = yf.Engine()
        engine.load(df)

        X = X.astype(np.float64)
        x_shape = (1, X.shape[1])

        for i, x in enumerate(X):
            x = x.reshape(x_shape)

            feats = engine.processAudio(x)

            ## if i == 0:
            ##     import IPython
            ##     IPython.embed()

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
                if fx_group == 'melspec':
                    # log melspec features
                    fxs = np.log10(fxs)
                X_prime[i, offset:(offset + fxs.shape[0])] = fxs
                offset += fxs.shape[0]

        return X_prime
