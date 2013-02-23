import numpy as np
from time import time
from sklearn.cross_validation import cross_val_score
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler

from transform import FlattenTransformer
from transform import SpectrogramTransformer
from transform import SpectrogramStatsTransformer
from ranking import RankSVM
from ranking import SVMPerf
from ranking import RGradientBoostingClassifier

from sklearn.grid_search import GridSearchCV

import IPython

data = np.load("data/train.npz")

X = data["X_train"]
y = data["y_train"]

clf = GradientBoostingClassifier(n_estimators=500, max_depth=3,
                                 min_samples_leaf=3, learning_rate=0.1)
## clf = ExtraTreesClassifier(n_estimators=100, min_samples_split=1, max_features=None)
# optimal C for raw w/o scaling is 0.00001
# optimal C for stats w/o scaling is 0.00001
clf = LinearSVC(C=0.00001, loss='l1', dual=True, class_weight='auto')
## clf = RGradientBoostingClassifier({"distribution": "auc",
##                                    "shrinkage": 0.1,
##                                    "n.tree": 200,
##                                    "bag.fraction": 1.0,
##                                    "verbose": True,
##                                    "n.minobsinnode": 1,
##                                    "interaction.depth": 2,
##                                    "offset": False})

## clf = SVMPerf(C=0.00001, verbose=0, tol=0.001)
## clf = Pipeline(steps=[('scale', StandardScaler()),
##                       ('svm', clf)])
st = SpectrogramTransformer(NFFT=256, clip_upper=500, clip_lower=0,
                            noverlap=0.6, dtype=np.float64,
                            whiten=None, log=True, flatten=False)
X = st.fit_transform(X)

tf = FeatureUnion([
    ('spec', FlattenTransformer(scale=1.0)),
    ('sst1', SpectrogramStatsTransformer(axis=1)),
    ('sst0', SpectrogramStatsTransformer(axis=0)),
    ])

X = tf.fit_transform(X)

clf.fit(X, y)

## data = np.load("data/test.npz")
## X_test = data["X_test"]

## X_test = st.fit_transform(X_test)
## X_test = tf.fit_transform(X_test)

## y_scores = clf.decision_function(X_test)
## np.savetxt("linsvm1.txt", y_scores)

print X.shape
print st
print clf
scores = cross_val_score(clf, X, y, scoring="roc_auc", cv=5, n_jobs=2)
print("spectrogram: %.5f (%.5f)" % (scores.mean(), scores.std()))

## param_grid = {'C': [0.0001, 0.00001, 0.000001, 0.0000001],
##               }

## grid_search = GridSearchCV(clf, param_grid, scoring='roc_auc', n_jobs=2, iid=True,
##                            refit=True, cv=5, verbose=0, pre_dispatch='2*n_jobs')


## print('_' * 80)
## t0 = time()
## grid_search.fit(X, y)
## print("done in %0.3fs" % (time() - t0))
## print

## print("Best score: %0.3f" % grid_search.best_score_)
## print("Best parameters set:")
## best_parameters = grid_search.best_estimator_.get_params()
## for param_name in sorted(param_grid.keys()):
##     print("\t%s: %r" % (param_name, best_parameters[param_name]))


## print "done"
## clf = SGDClassifier(n_iter=10, alpha=0.001, loss='hinge',
##                     learning_rate='invscaling', eta0=0.1, power_t=0.1)
## clf = SGDClassifier()
## #clf = LinearSVC(C=1.0, loss='l1', dual=True)

## clf = Pipeline(steps=[('spectrogram',
##                        SpectrogramTransformer(NFFT=256, noverlap=200,
##                                               dtype=np.float32)),
## #                      ('whitening', PCA(n_components=300, whiten=False)),
##                       ('scale', StandardScaler()),
##                       ('sgd', clf)])

## param_grid = {'spectrogram__NFFT': [256],
##               'spectrogram__noverlap': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
##               'spectrogram__clip': [750, 500, 250]}

## grid_search = GridSearchCV(clf, param_grid, scoring='roc_auc', n_jobs=2, iid=True,
##                            refit=True, cv=5, verbose=0, pre_dispatch='2*n_jobs')
## print('_' * 80)
## ## print clf
## ## print
## ## scores = cross_val_score(clf, X, y, scoring="roc_auc", cv=5)
## ## print "spectrogram =", scores.mean(), scores.std()

## t0 = time()
## grid_search.fit(X, y)
## print("done in %0.3fs" % (time() - t0))
## print

## print("Best score: %0.3f" % grid_search.best_score_)
## print("Best parameters set:")
## best_parameters = grid_search.best_estimator_.get_params()
## for param_name in sorted(param_grid.keys()):
##     print("\t%s: %r" % (param_name, best_parameters[param_name]))
