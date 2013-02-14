#! /usr/bin/env python
# -*- coding: utf-8 -*-

import aifc
import numpy as np
import os
import os.path
import struct


def load_aiff(filename):
    # Load data
    fd = aifc.open(filename, "r")
    sample_width = fd.getsampwidth()
    n_frames = fd.getnframes()
    n_channels = fd.getnchannels()
    data = fd.readframes(fd.getnframes())
    fd.close()

    # Convert bytes into Numpy array
    samples = np.fromstring(data, np.short).byteswap()

    return samples


def load_training_data(file_labels, dir_aiff):
    X = []
    y = []

    fd_labels = open(file_labels, "r")
    for line in fd_labels:
        filename, label = line.strip().split(",")
        X.append(load_aiff(os.path.join(dir_aiff, filename)))
        y.append(int(label))
    fd_labels.close()

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int)


def load_test_data(dir_aiff, n=54503):
    X = []

    for i in xrange(1, n+1):
        filename = "test%d.aiff" % i
        X.append(load_aiff(os.path.join(dir_aiff, filename)))

    return np.array(X, dtype=np.float32)


def yaafe_features(samples):
    from yaafelib import FeaturePlan, Engine

    size = samples.shape[1] # Is that a good stepSize?

    fp = FeaturePlan(sample_rate=2000)
    fp.addFeature("ampl: AmplitudeModulation")
    fp.addFeature("autocorrel: AutoCorrelation stepSize=%d" % size)
    fp.addFeature("complex: ComplexDomainOnsetDetection stepSize=%d" % size)
    fp.addFeature("energy: Energy stepSize=%d" % size)
    fp.addFeature("envelope: Envelope")
    fp.addFeature("envelopestats: EnvelopeShapeStatistics")
    fp.addFeature("lpc: LPC stepSize=%d" % size)
    fp.addFeature("loudness: Loudness stepSize=%d" % size)
    fp.addFeature("magspec: MagnitudeSpectrum stepSize=%d" % size)
    fp.addFeature("melspec: MelSpectrum stepSize=%d" % size)
    fp.addFeature("obsi: OBSI stepSize=%d" % size)
    fp.addFeature("obsir: OBSIR stepSize=%d" % size)
    fp.addFeature("perceptualsharpness: PerceptualSharpness stepSize=%d" % size)
    fp.addFeature("perceptualspread: PerceptualSpread stepSize=%d" % size)
    fp.addFeature("scfpb: SpectralCrestFactorPerBand stepSize=%d" % size)
    fp.addFeature("sd: SpectralDecrease stepSize=%d" % size)
    fp.addFeature("sf: SpectralFlatness stepSize=%d" % size)
    fp.addFeature("sfpb: SpectralFlatnessPerBand stepSize=%d" % size)
    fp.addFeature("sflux: SpectralFlux stepSize=%d" % size)
    fp.addFeature("srolloff: SpectralRolloff stepSize=%d" % size)
    fp.addFeature("sss: SpectralShapeStatistics stepSize=%d" % size)
    fp.addFeature("sslope: SpectralSlope stepSize=%d" % size)
    fp.addFeature("tss: TemporalShapeStatistics stepSize=%d" % size)
    fp.addFeature("zcr: ZCR stepSize=%d" % size)

    df = fp.getDataFlow()
    engine = Engine()
    engine.load(df)

    all_features = []

    for s in samples:
        features = engine.processAudio(s.astype("float64").reshape(1, size))
        features = np.hstack(features[k] for k in sorted(features.keys()))
        features = features.astype("float32")
        features[np.isnan(features)] = -10e10 # Better treatment for NaN and Inf?
        features[np.isinf(features)] = 10e10
        features = features.flatten()

        all_features.append(features)

    return np.array(all_features)


if __name__ == "__main__":
    X_train, y_train = load_training_data("data/train.csv", "data/train")
    X_test = load_test_data("data/test")

    # Save for later as numpy arrays
    fd = open("data/train.npz", "wb")
    np.savez(fd, X_train=X_train, y_train=y_train)
    fd.close()

    fd = open("data/test.npz", "wb")
    np.savez(fd, X_test=X_test)
    fd.close()

