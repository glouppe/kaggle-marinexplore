#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.metrics.scorer import auc_scorer

class MagicTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, magic_column=0):
        self.magic_column = magic_column

    def fit(self, X, y=None, **fit_args):
        self.positions_ = X[:, self.magic_column]
        self.y_ = y
        return self

    def transform(self, X):
        all_features = []

        for interval in [0.1, 0.05, 0.01, 0.005, 0.001]:
            avg_before = np.zeros(X.shape[0])
            avg_after = np.zeros(X.shape[0])

            for i, pos in enumerate(X[:, self.magic_column]):
                m = np.mean(self.y_[(self.positions_ > pos - interval) & (self.positions_ < pos)])
                if np.isnan(m) or np.isinf(m):
                    avg_before[i] = 0.5
                else:
                    avg_before[i] = m
                
                m = np.mean(self.y_[(self.positions_ < pos + interval) & (self.positions_ > pos)])
                if np.isnan(m) or np.isinf(m):
                    avg_after[i] = 0.5
                else:
                    avg_after[i] = m

            all_features.append(avg_before)
            all_features.append(avg_after)
            all_features.append(avg_before+avg_after)

        return np.hstack((X, np.array(all_features).T))


def load_data(prefix="train"):
    if prefix == "train":
        n = 30000
    else:
        n = 54503

    X = np.hstack([v.reshape((-1, 1)) for v in [
        np.arange(n, dtype=np.float) / n,
        np.loadtxt("stacks/adaboost-500-%s.txt" % prefix),
        #np.loadtxt("stacks/adaboost-500-magic-%s.txt" % prefix),
        np.loadtxt("stacks/rf-1000-%s.txt" % prefix),
        #np.loadtxt("stacks/rf-1000-magic-%s.txt" % prefix),
        np.loadtxt("stacks/et-500-%s.txt" % prefix),
        #np.loadtxt("stacks/et-500-magic-%s.txt" % prefix),
        #np.loadtxt("stacks/dbn-spec-100-%s.txt" % prefix),
        np.loadtxt("stacks/dbn-500-500-250-%s.txt" % prefix),
        np.loadtxt("stacks/dbn-2000-%s.txt" % prefix),
        np.loadtxt("stacks/my-mfcc-%s.txt" % prefix),
        np.loadtxt("stacks/knn-1800-25-%s.txt" % prefix),
        np.loadtxt("stacks/linearsvc-8000-11-L1-L2-%s.txt" % prefix),
        #np.loadtxt("stacks/linearsvc-8000-11-L1-L2-magic-%s.txt" % prefix),
        np.mean(np.hstack([np.loadtxt("stacks/dbn-1200-%d-%s.txt" % (i, prefix)).reshape((-1, 1))  for i in range(1, 21)]), axis=1),
        np.mean(np.hstack([np.loadtxt("stacks/gbrt-500-%d-%s.txt" % (i, prefix)).reshape((-1, 1))  for i in range(1, 21)]), axis=1),
        np.mean(np.hstack([np.loadtxt("stacks/gbrt-500-old-%d-%s.txt" % (i, prefix)).reshape((-1, 1))  for i in range(1, 21)]), axis=1),
        np.mean(np.hstack([np.loadtxt("stacks/gbrt-2500-%d-%s.txt" % (i, prefix)).reshape((-1, 1))  for i in range(1, 6)]), axis=1),
        np.mean(np.hstack([np.loadtxt("stacks/gbrt-2500-old-%d-%s.txt" % (i, prefix)).reshape((-1, 1))  for i in range(1, 11)]), axis=1),
        np.mean(np.hstack([np.loadtxt("stacks/gbrt-2500-magic-%d-%s.txt" % (i, prefix)).reshape((-1, 1))  for i in range(1, 21)]), axis=1),
    ]])

    if prefix == "train":
        data = np.load("data/train.npz")
        y = data["y_train"]

        return X, y

    else:
        return X, None

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.grid_search import GridSearchCV

X_train, y_train = load_data("train")

pipe = Pipeline([
        ("magic", MagicTransformer()),
        ("clf", GradientBoostingClassifier(random_state=13))
    ])

params = {
    "clf__n_estimators": [1000],
    "clf__max_depth": [5],
    "clf__min_samples_split": [28, 29, 30],
    "clf__learning_rate": np.linspace(0.013, 0.015, num=5), #0.010888],
    "clf__max_features": [0.1, 0.15, 0.2],
    "clf__subsample": [0.895] #np.linspace(0.89, 0.9, num=3)
}

cv = KFold(X_train.shape[0], 3, shuffle=True, random_state=13)
clf = GridSearchCV(pipe, params, cv=cv, scoring="roc_auc", verbose=3, n_jobs=12)
clf.fit(X_train, y_train)

print clf.best_score_
print clf.best_params_
#print clf.best_estimator_.feature_importances_


X_test, _ = load_data("test")

np.savetxt("stacking15.txt", clf.best_estimator_.decision_function(X_test)[:, 0])

decisions = np.zeros(X_test.shape[0])

for i in range(10):
    print "Estimator %d" % i
    c = clf.best_estimator_
    c.set_params(clf__random_state=i)
    c.fit(X_train, y_train)
    decisions += c.decision_function(X_test)[:, 0]

np.savetxt("stacking15-all.txt", decisions)

