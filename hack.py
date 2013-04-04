import numpy as np
from scipy import signal

from time import time
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler

from transform import FlattenTransformer
from transform import SpectrogramTransformer
from transform import StatsTransformer
from transform import FilterTransformer
from transform import TemplateMatcher
from transform import mfcc
from ranking import RankSVM
from ranking import SVMPerf
#from ranking import RGradientBoostingClassifier

from sklearn.grid_search import GridSearchCV

import IPython

data = np.load("data/train.npz")

X = data["X_train"]
y = data["y_train"]

ind = np.arange(y.shape[0])

X_train, X_test, y_train, y_test, ind_train, ind_test = train_test_split(
            X, y, ind, test_size=0.5, random_state=42)


## clf = GradientBoostingClassifier(n_estimators=250, max_depth=2,
##                                  min_samples_leaf=17, learning_rate=0.1,
##                                  verbose=0)

# optimal C for raw w/o scaling is 0.00001
# optimal C for stats w/o scaling is 0.00001
clf = LinearSVC(C=0.001, loss='l1', dual=True)



## clf = SVMPerf(C=0.00000001, verbose=0, tol=0.001)
clf = Pipeline(steps=[
    ('scale', StandardScaler()),
    ('svm', clf)
    ])

fu = FeatureUnion([
        #('spec', FlattenTransformer(scale=1.0)),
        ('st1', StatsTransformer(axis=1)),
        #('st0', StatsTransformer(axis=0))
    ])

tf = Pipeline(steps=[('specg', SpectrogramTransformer(NFFT=256, clip=500,
                                               noverlap=0.5,
                                               dtype=np.float32,
                                               log=False, flatten=False)),
                     ('tm', TemplateMatcher(raw=False)),
                     #('flatten', FlattenTransformer()),
                     #('fu', fu),
                     ])

print('_' * 80)
print('transforming data')
print

X_train = tf.fit_transform(X_train, y=y_train)
X_test = tf.transform(X_test)

print X_train.shape

#IPython.embed()

## param_grid = {'svm__C': [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001],
##               }

## grid_search = GridSearchCV(clf, param_grid, scoring='roc_auc', n_jobs=2, iid=True,
##                            refit=True, cv=3, verbose=0)

## print('_' * 80)
## t0 = time()
## grid_search.fit(X_train, y_train)
## print("done in %0.3fs" % (time() - t0))
## print

## print("Best score: %0.3f" % grid_search.best_score_)
## print("Best parameters set:")
## best_parameters = grid_search.best_estimator_.get_params()
## for param_name in sorted(param_grid.keys()):
##     print("\t%s: %r" % (param_name, best_parameters[param_name]))

print('_' * 80)
print('training best parameters on all train data')
print
#clf.set_params(**best_parameters)
clf.fit(X_train, y_train)

from sklearn.metrics import auc_score

y_scores = clf.decision_function(X_test).ravel()
print "AUC: %.6f" % auc_score(y_test, y_scores)
