#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
import sys

from scipy.sparse import csr_matrix
from scipy.stats.mstats import mode

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.externals.joblib import Parallel, delayed, cpu_count
from sklearn.utils import check_random_state

from _estimator import leaf_transform

MAX_INT = np.iinfo(np.int32).max


def _random_window(clip, size, random_state=None):
    random_state = check_random_state(random_state)

    w = np.zeros(2 * size)
    start = random_state.randint(0, len(clip) - size)
    w[:size] = clip[start:start+size]
    w[size:] = abs(np.fft.fft(w[:size]))

    return w


def _partition_clips(n_jobs, n_clips):
    if n_jobs == -1:
        n_jobs = min(cpu_count(), n_clips)

    else:
        n_jobs = min(n_jobs, n_clips)

    counts = [n_clips / n_jobs] * n_jobs

    for i in xrange(n_clips % n_jobs):
        counts[i] += 1

    starts = [0] * (n_jobs + 1)

    for i in xrange(1, n_jobs + 1):
        starts[i] = starts[i - 1] + counts[i - 1]

    return n_jobs, counts, starts


def _parallel_make_subwindows(X, y, dtype, n_subwindows, size, seed):
    random_state = check_random_state(seed)

    size = int(size * X.shape[1])
    _X = np.zeros((len(X) * n_subwindows, 2 * size), dtype=dtype)
    _y = np.zeros((len(X) * n_subwindows), dtype=np.int32)

    i = 0

    for clip, target in zip(X, y):
        for w in xrange(n_subwindows):
            _X[i, :] = _random_window(clip, size, random_state=random_state)
            _y[i] = target
            i += 1

    return _X, _y


class PyxitClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator,
                       n_subwindows=10,
                       size=0.25,
                       n_jobs=1,
                       random_state=None,
                       verbose=0):
        self.base_estimator = base_estimator
        self.n_subwindows = n_subwindows
        self.size = size
        self.n_jobs = n_jobs
        self.random_state = check_random_state(random_state)
        self.verbose = verbose

        self.maxs = None

    def extract_subwindows(self, X, y, dtype=np.float32):
        # Assign chunk of subwindows to jobs
        n_jobs, _, starts = _partition_clips(self.n_jobs, len(X))

        # Parallel loop
        if self.verbose > 0:
            print "[estimator.PyxitClassifier.extract_subwindows] Extracting random subwindows"

        all_data = Parallel(n_jobs=n_jobs)(
            delayed(_parallel_make_subwindows)(
                X[starts[i]:starts[i + 1]],
                y[starts[i]:starts[i + 1]],
                dtype,
                self.n_subwindows,
                self.size,
                self.random_state.randint(MAX_INT))
            for i in xrange(n_jobs))

        # Reduce
        _X = np.vstack(X for X, _ in all_data)
        _y = np.concatenate([y for _, y in all_data])

        return _X, _y

    def extend_mask(self, mask):
        mask_t = np.zeros(len(mask) * self.n_subwindows, dtype=np.int)

        for i in xrange(len(mask)):
            offset = mask[i] * self.n_subwindows

            for j in xrange(self.n_subwindows):
                mask_t[i * self.n_subwindows + j] = offset + j

        return mask_t

    def fit(self, X, y, _X=None, _y=None):
        # Collect some data
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        y = np.searchsorted(self.classes_, y)

        # Extract subwindows
        if _X is None or _y is None:
            _X, _y = self.extract_subwindows(X, y)

        # Fit base estimator
        if self.verbose > 0:
            print "[estimator.PyxitClassifier.fit] Building base estimator"

        self.base_estimator.fit(_X, _y)

        return self

    def predict(self, X, _X=None):
        return self.classes_.take(
            np.argmax(self.predict_proba(X, _X), axis=1),  axis=0)

    def predict_proba(self, X, _X=None):
        # Extract subwindows
        if _X is None:
            y = np.zeros(X.shape[0])
            _X, _y = self.extract_subwindows(X, y)

        # Predict proba
        if self.verbose > 0:
            print "[estimator.PyxitClassifier.predict_proba] Computing class probabilities"

        y = np.zeros((X.shape[0], self.n_classes_))
        inc = 1.0 / self.n_subwindows

        try:
            _y = self.base_estimator.predict_proba(_X)

            for i in xrange(X.shape[0]):
                y[i] = np.sum(_y[i * self.n_subwindows:(i + 1) * self.n_subwindows], axis=0) / self.n_subwindows

        except:
            _y = self.base_estimator.predict(_X)

            for i in xrange(X.shape[0]):
                for j in xrange(i * self.n_subwindows, (i + 1) * self.n_subwindows):
                    y[i, _y[j]] += inc

        return y

    def transform(self, X, _X=None):
        # Predict proba
        if self.verbose > 0:
            print "[estimator.PyxitClassifier.transform] Transforming into leaf features"

        # Extract subwindows
        if _X is None:
            y = np.zeros(X.shape[0])
            _X, _y = self.extract_subwindows(X, y)

        # Leaf transform
        row, col, data, node_count = leaf_transform([tree.tree_ for tree in self.base_estimator.estimators_], _X, X.shape[0], self.n_subwindows)
        __X = csr_matrix((data, (row, col)), shape=(X.shape[0], node_count), dtype=np.float)

        # TODO: Scale features from [0, max] to [0, 1]

        return __X
