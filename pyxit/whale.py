
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import auc_score
from sklearn.svm import SVC

from estimator import PyxitClassifier

train_file = sys.argv[1]
test_file = sys.argv[2]

n_subwindows = int(sys.argv[3])
size = float(sys.argv[4])
n_estimators = int(sys.argv[5])
min_samples_split = int(sys.argv[6])
n_jobs = int(sys.argv[7])

data = np.load(train_file)
X_train = data["X_train"]
y_train = data["y_train"]

# data = np.load(test_file)
# X_test = data["X_test"]

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.666)

print "[PyxitClassifier]"
clf = PyxitClassifier(n_subwindows=n_subwindows,
                     size=size,
                     base_estimator=ExtraTreesClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, n_jobs=n_jobs),
                     verbose=1,
                     n_jobs=n_jobs,
                     random_state=0)

print clf

clf.fit(X_train, y_train)

y_proba = clf.predict_proba(X_test)
np.savetxt("pyxit.txt", y_proba[:, 1])

print auc_score(y_test, y_proba[:, 1])
print

print "[PyxitClassifier+SVC]"
_X_train = clf.transform(X_train)
_X_test = clf.transform(X_test)

svc = SVC(probability=True, kernel="linear")
svc.fit(_X_train, y_train)

y_proba = svc.predict_proba(_X_test)
np.savetxt("pyxit-svc.txt", y_proba[:, 1])

print auc_score(y_test, svc.predict_proba(_X_test)[:, 1])
