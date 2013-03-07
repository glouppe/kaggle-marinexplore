#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys

from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics.scorer import auc_scorer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import normalize, scale
from scipy.io import loadmat

from transform import FlattenTransformer
from transform import SpectrogramTransformer
from transform import StatsTransformer
from transform import FuncTransformer


def load_data(full=False):
    tf = Pipeline([
            #("func", FuncTransformer(normalize, axis=1, norm="l2", copy=False)),
            ("union", FeatureUnion([
                ('spec', FlattenTransformer(scale=1.0)),
                ('st1', StatsTransformer(axis=1)),
                ('st0', StatsTransformer(axis=0))]))
        ])

    datasets = [
        ("ceps_2000", 198, 9),
        ("specs_2000", 198, 9),
        ("ceps_4000", 98, 9),
        ("specs_4000", 98, 13),
        ("ceps_8000", 48, 9),
        ("specs_8000", 48, 17),
        ("ceps_16000", 23, 9),
        ("specs_16000", 23, 21),
        ("mfcc_8000", 48, 13),
        ("mfcc_16000", 23, 13),
        ("mfcc_32000", 11, 13),
        ("mfcc_64000", 4, 13)
    ]

    def _load(datasets, prefix="data/train_"):
        all_arrays = []

        for name, d0, d1 in datasets:
            data = loadmat("%s%s.mat" % (prefix, name))
            X = data[sorted(data.keys())[-1]]
            X = X.reshape((X.shape[0], d0, d1))
            X = tf.transform(X)
            all_arrays.append(X)

        X = np.hstack(all_arrays)

        return X

    if not full:
        data = np.load("data/train.npz")
        y = data["y_train"]
        X = _load(datasets, prefix="data/train_")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    else:
        data = np.load("data/train.npz")
        y_train = data["y_train"]
        X_train = _load(datasets, prefix="data/train_")
        X_test = _load(datasets, prefix="data/test_")
        y_test = None

    return X_train, X_test, y_train, y_test


def build_extratrees(argv, n_features):
    from sklearn.ensemble import ExtraTreesClassifier

    parameters = {
        "n_estimators": int(argv[0]),
        "max_features": int(argv[1]),
        "min_samples_split": int(argv[2]),
    }

    clf = ExtraTreesClassifier(**parameters)

    return clf


def build_randomforest(argv, n_features):
    from sklearn.ensemble import RandomForestClassifier

    parameters = {
        "n_estimators": int(argv[0]),
        "max_features": int(argv[1]),
        "min_samples_split": int(argv[2]),
    }

    clf = RandomForestClassifier(**parameters)

    return clf


def build_gbrt(argv, n_features):
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


def build_dbn(argv, n_features):
    from nolearn.dbn import DBN

    units = [n_features] + [int(n) for n in argv[0].split("-")] + [2]

    parameters = {
        "epochs": int(argv[1]),
        "learn_rates": float(argv[2]),
        "momentum": float(argv[3]),
        "verbose": 0
    }

    clf = DBN(units, **parameters)

    return clf


if __name__ == "__main__":
    argv = sys.argv[1:]
    print argv # log command line parameters

    # Load data
    print "Loading data..."
    X_train, X_test, y_train, y_test = load_data()
    print "X_train.shape =", X_train.shape
    print "y_train.shape =", y_train.shape

    # Estimator setup
    print "Estimator setup..."
    clf = locals()["build_%s" % argv[0]](argv[1:], n_features=X_train.shape[1])
    print clf

    # Estimator training
    print "Training..."
    clf.fit(X_train, y_train)

    # AUC
    if y_test is not None:
        print "AUC =", auc_scorer(clf, X_test, y_test)

    # Save predictions
    if y_test is None:
        y_pred = clf.predict_proba(X_test)
        np.savetxt(argv[-1], y_pred[:, 1])

