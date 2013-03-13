#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys

from sklearn.cross_validation import train_test_split, KFold
from sklearn.metrics.scorer import auc_scorer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import normalize, scale, StandardScaler
from scipy.io import loadmat

from transform import FlattenTransformer
from transform import SpectrogramTransformer
from transform import StatsTransformer
from transform import FuncTransformer
from transform.pool import PoolTransformer


def load_data(argv):
    tf = Pipeline([
            #("func", FuncTransformer(normalize, axis=1, norm="l2", copy=False)),
            ("union", FeatureUnion([
                ('spec', FlattenTransformer(scale=1.0)),
                #('max-pool', PoolTransformer(size=(2,2), step=(2,2))),
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
        ("ceps_32000", 11, 9),
        ("specs_32000", 11, 25),
        ("mfcc_8000", 48, 13),
        ("mfcc_16000", 23, 13),
        ("mfcc_32000", 11, 13),
        ("mfcc_64000", 4, 13),
        #("wiener1spectro", 65, 30),
    ]

    def _load(datasets, prefix="data/train_"):
        all_arrays = []

        for name, d0, d1 in datasets:
            data = loadmat("%s%s.mat" % (prefix, name))
            data.pop("__globals__")
            data.pop("__header__")
            data.pop("__version__")
            X = data[sorted(data.keys())[-1]]
            X = X.reshape((X.shape[0], d0, d1))
            X = tf.transform(X)
            all_arrays.append(X)

        X = np.hstack(all_arrays)
        print "X.shape =", X.shape

        return X

    try:
        n_features = int(argv[0]) # will fail if string; this is fine
        argv.pop(0)
        importances = np.loadtxt("feature-importances-rf-flat.txt")
        print "importances.shape =", importances.shape
        indices = np.argsort(importances)[::-1]
        indices = indices[:n_features]
    except:
        indices = None

    # Train set
    data = np.load("data/train.npz")
    y_train = data["y_train"]
    X_train = _load(datasets, prefix="data/train_")
    if indices is not None:
        X_train = X_train[:, indices]

    # Test set
    y_test = None
    X_test = _load(datasets, prefix="data/test_")
    if indices is not None:
        X_test = X_test[:, indices]

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
        "n_jobs": 4,
    }

    clf = RandomForestClassifier(**parameters)

    return clf


def build_gbrt(argv, n_features):
    from sklearn.ensemble import GradientBoostingClassifier

    parameters = {
        "n_estimators": int(argv[0]),
        "max_depth": int(argv[1]),
        "learning_rate": float(argv[2]),
        "max_features": float(argv[3]),
        "min_samples_split": int(argv[4]),
        "subsample": float(argv[5]),
        #"verbose": 3
    }

    clf = GradientBoostingClassifier(**parameters)

    return clf


def build_dbn(argv, n_features):
    """argv: units epochs epochs_pretrain learn_rates learn_rates_pretrain"""
    from nolearn.dbn import DBN

    units = [n_features] + [int(n) for n in argv[0].split("-")] + [2]
    n_layers = len(units) - 2

    learn_rates = eval(argv[3])
    learn_rates_pretrain = eval(argv[4])

    parameters = {
        "epochs": int(argv[1]),
        "epochs_pretrain": int(argv[2]),
        "learn_rates": learn_rates,
        "learn_rates_pretrain": learn_rates_pretrain,
        "l2_costs": 0.0,
        "l2_costs_pretrain": 0.0001,
        "momentum": 0.9,
        "verbose": 0,
        "real_valued_vis": True,
        "use_re_lu": False,
        "scales": 0.01,
        "minibatch_size": 200,
        "dropouts": [0.2] + [0.5] * n_layers,
        }

    dbn = DBN(units, **parameters)

    clf = Pipeline(steps=[('scale', StandardScaler()),
                          ('dbn', dbn)])

    return clf


if __name__ == "__main__":
    argv = sys.argv[1:]
    print argv # log command line parameters

    # Load data
    print "Loading data..."
    X_train, X_test, y_train, y_test = load_data(argv)
    print "X_train.shape =", X_train.shape
    print "y_train.shape =", y_train.shape

    # Estimator setup
    print "Estimator setup..."
    clf = locals()["build_%s" % argv[0]](argv[1:], n_features=X_train.shape[1])
    print clf

    # Build predictions on train set
    print "KFold..."

    y_train_proba = np.zeros(y_train.shape)

    for train, test in KFold(n=len(y_train), n_folds=3, random_state=42, shuffle=True):
        clf.fit(X_train[train], y_train[train])

        if hasattr(clf, "decision_function"):
            y_train_proba[test] = clf.decision_function(X_train[test])
        else:
            y_train_proba[test] = clf.predict_proba(X_train[test])[:, 1]

    np.savetxt("%s-train.txt" % argv[-1], y_train_proba)

    # Build predictions on test set
    print "Training..."
    clf.fit(X_train, y_train)

    if hasattr(clf, "decision_function"):
        y_test = clf.decision_function(X_test)
    else:
        y_test = clf.predict_proba(X_test)[:, 1]

    np.savetxt("%s-test.txt" % argv[-1], y_test)
