import numpy as np
from time import time
from sklearn.cross_validation import cross_val_score
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler

from transform import FlattenTransformer
from transform import SpectrogramTransformer
from transform import SpectrogramStatsTransformer
from ranking import RankSVM

from sklearn.grid_search import GridSearchCV

import IPython

data = np.load("data/train_small.npz")

X = data["X_train"]
y = data["y_train"]

## clf = GradientBoostingClassifier(n_estimators=500, max_depth=4,
##                                  min_samples_leaf=7, learning_rate=0.2)

clf = LinearSVC(C=0.00001, loss='l1', dual=True)
## clf = Pipeline(steps=[('scale', StandardScaler()),
##                       ('svm', clf)])
st = SpectrogramTransformer(NFFT=256, clip=500, noverlap=0.6, dtype=np.float64,
                            whiten=None, log=True, flatten=False)
X = st.fit_transform(X)

tf = FeatureUnion([
    ('spec', FlattenTransformer()),
    ('sst', SpectrogramStatsTransformer()),
    ])

X = tf.fit_transform(X)

print X.shape
print clf
scores = cross_val_score(clf, X, y, scoring="roc_auc", cv=5)
print("spectrogram: %.5f (%.5f)" % (scores.mean(), scores.std()))

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
