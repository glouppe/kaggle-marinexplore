#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics.scorer import auc_scorer
from sklearn.pipeline import FeatureUnion
from scipy.io import loadmat

from transform import FlattenTransformer
from transform import StatsTransformer
from transform.peaks import PeaksTransformer

def load_data(full=False):
    data = np.load("data/train.npz")
    y = data["y_train"]
    n_samples = len(y)

    tf = FeatureUnion([
        ('spec', FlattenTransformer(scale=1.0)),
        ('st1', StatsTransformer(axis=1)),
        ('st0', StatsTransformer(axis=0)),
    ])

    data = loadmat("data/train_specs.mat")
    X_specs = data["train_specs"]
    X_specs = X_specs.reshape((n_samples, 98, 13))
    X_specs = tf.transform(X_specs)

    data = loadmat("data/train_ceps.mat")
    X_ceps = data["train_ceps"]
    X_ceps = X_ceps.reshape((n_samples, 98, 9))
    X_ceps = tf.transform(X_ceps)

    data = loadmat("data/train_mfcc.mat")
    X_mfcc = data["train_mfcc"]
    X_mfcc = X_mfcc.reshape((n_samples, 23, 13))
    X_mfcc = tf.transform(X_mfcc)

    print('X_specs: %s' % str(X_specs.shape))
    print('X_ceps: %s' % str(X_ceps.shape))
    print('X_mfcc: %s' % str(X_mfcc.shape))
    X = np.hstack((X_specs, X_ceps, X_mfcc))

    if full:
        X_train = X
        y_train = y

        data = np.load("data/test.npz")
        X_test = data["X_test"]
        n_samples = X_test.shape[0]

        data = loadmat("data/test_specs.mat")
        X_specs = data["test_specs"]
        X_specs = X_specs.reshape((n_samples, 98, 13))
        X_specs = tf.transform(X_specs)

        data = loadmat("data/test_ceps.mat")
        X_ceps = data["test_ceps"]
        X_ceps = X_ceps.reshape((n_samples, 98, 9))
        X_ceps = tf.transform(X_ceps)

        data = loadmat("data/test_mfcc.mat")
        X_mfcc = data["test_mfcc"]
        X_mfcc = X_mfcc.reshape((n_samples, 23, 13))
        X_mfcc = tf.transform(X_mfcc)

        X_test = np.hstack((X_specs, X_ceps, X_mfcc))
        y_test = None
    else:

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                            random_state=42)

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = load_data(full=True)

clf = GradientBoostingClassifier(n_estimators=1000, max_depth=5,
                                 learning_rate=0.1, max_features=256,
                                 min_samples_split=7, verbose=3,
                                 random_state=13)

clf.fit(X_train, y_train)
if y_test is not None:
    from sklearn.metrics import auc_score
    print clf
    print "AUC: %.6f" % auc_score(y_test, clf.decision_function(X_test))

np.savetxt("gbrt2.txt", clf.decision_function(X_test))
