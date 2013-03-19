#! /usr/bin/env python
# -*- coding: utf-8 -*-
from time import time

import numpy as np

from scipy.io import loadmat
from scipy import signal

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import auc_score
from nolearn.dbn import DBN

from transform import SpectrogramTransformer
from transform import StatsTransformer
from transform import FilterTransformer
from transform import DiffTransformer
from transform import FlattenTransformer

import IPython


def chunked_predict_proba(dbn, X, n_chunks=10):
    """Runs ``dbn.predict_proba`` on chunks of X.

    This is handy to limit GPU memory consumption.

    Returns
    -------
    act : np.ndarray, shape = (X.shape[0], n_hidden_units)
        The activations of the last hidden layer of ``dbn``.
    """
    proba = np.empty((X.shape[0],), dtype=np.float64)
    start = 0
    chunk_size = int(np.ceil(X.shape[0] / float(n_chunks)))
    print "chunk_size: ", chunk_size
    for i in range(n_chunks):
        end = start + chunk_size
        proba[start:end] = dbn.predict_proba(X[start:end])[:, 1]
        start = end
    return proba


def chunked_transform(dbn, X, n_chunks=10, n_hidden_units=250):
    """Runs ``dbn.transform`` on chunks of X.

    This is handy to limit GPU memory consumption.

    Returns
    -------
    act : np.ndarray, shape = (X.shape[0], n_hidden_units)
        The activations of the last hidden layer of ``dbn``.
    """
    act = np.empty((X.shape[0], n_hidden_units), dtype=np.float64)
    start = 0
    chunk_size = int(np.ceil(X.shape[0] / float(n_chunks)))
    print "chunk_size: ", chunk_size
    for i in range(n_chunks):
        end = start + chunk_size
        act[start:end] = dbn.transform(X[start:end])
        start = end
    return act


def load_data(full=False):
    fu_tf = FeatureUnion([
        ('flatten', FlattenTransformer()),
        ('st1', StatsTransformer(axis=1)),
        ('st0', StatsTransformer(axis=0)),
    ])
    transformer = Pipeline([("spec", SpectrogramTransformer(flatten=False,
                                                            clip=750.0,
                                                            Fs=2000,
                                                            noverlap=0.5)),
                            ("wiener2", FilterTransformer(signal.wiener)),
                            ("fu", fu_tf),
                            #("norm", Normalizer(norm='l2')),
                            ("scale", StandardScaler()),
                            ])

    if not full:
        data = np.load("data/train.npz")
        X = data["X_train"]
        y = data["y_train"]
        n_samples = y.shape[0]
        ind = np.arange(n_samples)

        # Split into train/test
        X_train, X_test, y_train, y_test, ind_train, ind_test = train_test_split(
            X, y, ind, test_size=0.5, random_state=42)

        X_train = transformer.fit_transform(X_train, y_train)
        X_test = transformer.transform(X_test)

    else:
        data = np.load("data/train.npz")
        X_train = data["X_train"]
        y_train = data["y_train"]
        ind_train = np.arange(y_train.shape[0])

        transformer.fit(X_train, y_train)
        X_train = transformer.transform(X_train)

        data = np.load("data/test.npz")
        X_test = transformer.transform(data["X_test"])
        y_test = None
        ind_test = None

    return X_train, X_test, y_train, y_test, ind_train, ind_test


def load_data_mfcc(full=False, shuffle_train=True):
    data = np.load("data/train.npz")
    y = data["y_train"]
    n_samples = len(y)

    tf = FeatureUnion([
        ('flatten', FlattenTransformer()),
        ('st1', StatsTransformer(axis=1)),
        ('st0', StatsTransformer(axis=0)),
    ])

    mfcc_tf = FeatureUnion([
        ('flatten', FlattenTransformer()),
        ('st1', StatsTransformer(axis=1)),
        ('st0', StatsTransformer(axis=0)),
        #('diff1', DiffTransformer(n=1, axis=1, flatten=True)),
        #('diff2', DiffTransformer(n=2, axis=1, flatten=True)),
    ])

    specs_tf = clone(tf)
    data = loadmat("data/train_specs.mat")
    X_specs = data["train_specs"]
    X_specs = X_specs.reshape((n_samples, 98, 13))
    X_specs = specs_tf.fit_transform(X_specs, y)

    ceps_tf = clone(tf)
    data = loadmat("data/train_ceps.mat")
    X_ceps = data["train_ceps"]
    X_ceps = X_ceps.reshape((n_samples, 98, 9))
    X_ceps = ceps_tf.fit_transform(X_ceps, y)

    data = loadmat("data/train_mfcc.mat")
    X_mfcc = data["train_mfcc"]
    X_mfcc = X_mfcc.reshape((n_samples, 23, 13))
    X_mfcc = mfcc_tf.fit_transform(X_mfcc, y)

    ## raw_fu = FeatureUnion([
    ##     ('flatten', FlattenTransformer()),
    ##     ('st1', StatsTransformer(axis=1)),
    ##     ('st0', StatsTransformer(axis=0)),
    ## ])
    ## raw_tf = Pipeline([("spec", SpectrogramTransformer(flatten=False,
    ##                                                    clip=500.0,
    ##                                                    noverlap=0.5)),
    ##                    ("wiener2", FilterTransformer(signal.wiener)),
    ##                    ("fu", raw_fu),
    ##                    ])

    ## data = np.load("data/train.npz")
    ## X = data["X_train"]
    ## X_raw_specs = raw_tf.fit_transform(X)

    print('X_specs: %s' % str(X_specs.shape))
    print('X_ceps: %s' % str(X_ceps.shape))
    print('X_mfcc: %s' % str(X_mfcc.shape))
    X = np.hstack((X_specs, X_ceps, X_mfcc))  # , X_raw_specs))

    ind = np.arange(y.shape[0])

    if full:
        X_train = X
        y_train = y

        data = np.load("data/test.npz")
        X_test = data["X_test"]
        n_samples = X_test.shape[0]

        data = loadmat("data/test_specs.mat")
        X_specs = data["test_specs"]
        X_specs = X_specs.reshape((n_samples, 98, 13))
        X_specs = specs_tf.transform(X_specs)

        data = loadmat("data/test_ceps.mat")
        X_ceps = data["test_ceps"]
        X_ceps = X_ceps.reshape((n_samples, 98, 9))
        X_ceps = ceps_tf.transform(X_ceps)

        data = loadmat("data/test_mfcc.mat")
        X_mfcc = data["test_mfcc"]
        X_mfcc = X_mfcc.reshape((n_samples, 23, 13))
        X_mfcc = mfcc_tf.transform(X_mfcc)

        X_test = np.hstack((X_specs, X_ceps, X_mfcc))
        y_test = None
        ind_test = None
        ind_train = ind

        if shuffle_train:
            X_train, y_train, ind_train = shuffle(X_train, y_train, ind_train,
                                                  random_state=42)
    else:

        # Split into train/test
        X_train, X_test, y_train, y_test, ind_train, ind_test = train_test_split(
            X, y, ind, test_size=0.5, random_state=42)

    return X_train, X_test, y_train, y_test, ind_train, ind_test

X_train, X_test, y_train, y_test, ind_train, ind_test = load_data(full=False)

print('n_features: %d' % X_train.shape[1])


def fine_tune_callback(clf, epoch):
    if epoch % 10 == 0:
        y_scores = chunked_predict_proba(clf, X_test, n_chunks=100)
        print "AUC: %.6f" % auc_score(y_test, y_scores)


gdahl_parameters = {
    "epochs": 200,
    "epochs_pretrain": 100,
    "learn_rates_pretrain": [0.01, 0.01, 0.01],
    "learn_rates": 1.0,
    "l2_costs_pretrain": 0.000001,
    "l2_costs": 0.0,
    "momentum": 0.5,
    "momentum_pretrain": 0.5,
    "verbose": 2,
    "real_valued_vis": True,
    "use_re_lu": False,
    "scales": 0.01,
    "minibatch_size": 200,
    "dropouts": [0.2, 0.5, 0.5],
    #"output_act_funct": "Sigmoid",
    "fine_tune_callback": fine_tune_callback,
}

gdahl_units = [X_train.shape[1]] + [100, 100] + [2]
dbn = DBN(gdahl_units, **gdahl_parameters)


clf = dbn
## clf = Pipeline(steps=[
##     ('scale', StandardScaler()),
##     ('dbn', dbn),
##     ])

t0 = time()
clf.fit(X_train, y_train)
print('clf.fit took %ds' % (time() - t0))
#X_train_ = dbn.transform(X_train)
#X_test_ = dbn.transform(X_test)
#X_train_ = X_train
#X_test_ = X_test


#print "AUC: %.6f" % auc_score(y_test, clf.decision_function(X_test))
#y_scores = clf.predict_proba(X_test)[:, 1]
y_scores = chunked_predict_proba(clf, X_test, n_chunks=100)
if y_test is not None:
    print "AUC: %.6f" % auc_score(y_test, y_scores)
else:
    print('model trained!')


## np.savetxt("dbn3.txt", y_scores)


## X_train_ = chunked_transform(clf, X_train, n_chunks=100)
## X_test_ = chunked_transform(clf, X_test, n_chunks=100)

## ind_train_inv = ind_train.argsort()
## ind_test_inv = ind_test.argsort()
## X_train_ = X_train_[ind_train_inv]
## X_test_ = X_test_[ind_test_inv]

## np.savez('data/dbn_act_internal.npz', X_train=X_train_, X_test_=X_test_)

IPython.embed()
