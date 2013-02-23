#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys

from sklearn.cross_validation import train_test_split
from sklearn.metrics.scorer import auc_scorer
from sklearn.pipeline import FeatureUnion
from scipy.io import loadmat

from transform import FlattenTransformer
from transform import StatsTransformer


def load_data():
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

    X = np.hstack((X_specs, X_ceps, X_mfcc))

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                        random_state=42)

    return X_train, X_test, y_train, y_test


def build_extratrees(argv):
    from sklearn.ensemble import ExtraTreesClassifier

    parameters = {
        "n_estimators": int(argv[0]),
        "max_features": int(argv[1]),
        "min_samples_split": int(argv[2]),
    }

    clf = ExtraTreesClassifier(**parameters)

    return clf


def build_gbrt(argv):
    from sklearn.ensemble import GradientBoostingClassifier

    parameters = {
        "n_estimators": int(argv[0]),
        "max_depth": int(argv[1]),
        "learning_rate": float(argv[2]),
        "max_features": int(argv[3]),
        "min_samples_split": int(argv[4]),
    }

    clf = GradientBoostingClassifier(**parameters)

    return clf


if __name__ == "__main__":
    # Load data
    print "Loading data..."
    X_train, X_test, y_train, y_test = load_data()

    # Estimator setup
    print "Estimator setup..."
    argv = sys.argv
    clf = locals()["build_%s" % argv[1]](argv[2:])
    print clf # log the estimator parameters

    # Estimator training
    print "Training..."
    clf.fit(X_train, y_train)

    # AUC
    print "AUC =", auc_scorer(clf, X_test, y_test)
