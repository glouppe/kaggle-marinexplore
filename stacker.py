#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.metrics.scorer import auc_scorer


def load_data(prefix="train"):
    X = np.hstack([v.reshape((-1, 1)) for v in [
        np.loadtxt("stacks/adaboost-500-%s.txt" % prefix),
        np.loadtxt("stacks/rf-1000-%s.txt" % prefix),
        np.loadtxt("stacks/et-500-%s.txt" % prefix),
        np.loadtxt("stacks/dbn-500-500-250-%s.txt" % prefix),
        np.mean(np.hstack([np.loadtxt("stacks/gbrt-500-%d-%s.txt" % (i, prefix)).reshape((-1, 1))  for i in range(1, 21)]), axis=1),
        np.mean(np.hstack([np.loadtxt("stacks/gbrt-500-old-%d-%s.txt" % (i, prefix)).reshape((-1, 1))  for i in range(1, 21)]), axis=1),
        np.mean(np.hstack([np.loadtxt("stacks/gbrt-2500-%d-%s.txt" % (i, prefix)).reshape((-1, 1))  for i in range(1, 6)]), axis=1),
        np.mean(np.hstack([np.loadtxt("stacks/gbrt-2500-old-%d-%s.txt" % (i, prefix)).reshape((-1, 1))  for i in range(1, 11)]), axis=1),
    ]])

    if prefix == "train":
        data = np.load("data/train.npz")
        y = data["y_train"]

        return X, y

    else:
        return X, None


from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.grid_search import GridSearchCV

X_train, y_train = load_data("train")
#uniques = np.loadtxt("train-uniques.txt").astype(np.int)
#X_train = X_train[uniques]
#y_train = y_train[uniques]

params = {
    "n_estimators": [500],
    "max_depth": [4],
    "subsample": [0.95],
    "learning_rate": np.linspace(0.0005, 0.005, num=10),
    "max_features": [6, 7, 8]
}

clf = GridSearchCV(GradientBoostingClassifier(), params, cv=3, scoring="roc_auc", verbose=3, n_jobs=12)
clf.fit(X_train, y_train)

print clf.best_score_
print clf.best_params_

X_test, _ = load_data("test")
decisions = np.zeros(X_test.shape[0]) 

for i in range(50):
    print "Estimator %d" % i
    c = GradientBoostingClassifier(**clf.best_params_)
    c.fit(X_train, y_train)
    decisions += c.decision_function(X_test)[:, 0]
    
np.savetxt("stacking6.txt", decisions)

