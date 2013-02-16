import numpy as np
from time import time
from sklearn.cross_validation import cross_val_score
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transform import SpectrogramTransformer

from sklearn.grid_search import GridSearchCV

import IPython

data = np.load("data/train_small.npz")

X = data["X_train"]
y = data["y_train"]


print "done"
clf = SGDClassifier(n_iter=10, alpha=0.001, loss='hinge',
                    learning_rate='invscaling', eta0=0.1, power_t=0.1)
clf = SGDClassifier()
#clf = LinearSVC(C=1.0, loss='l1', dual=True)
#clf = SVC(C=.0001, kernel='linear')

clf = Pipeline(steps=[('spectrogram',
                       SpectrogramTransformer(NFFT=256, noverlap=200, clip=1.0,
                                              dtype=np.float32)),
#                      ('whitening', PCA(n_components=300, whiten=False)),
                      ('scale', StandardScaler()),
                      ('sgd', clf)])

param_grid = {'spectrogram__NFFT': [256],
              'spectrogram__noverlap': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
              'spectrogram__clip': [1.0, 0.75, 0.5, 0.25, 0.125]}

grid_search = GridSearchCV(clf, param_grid, scoring='roc_auc', n_jobs=3, iid=True,
                           refit=True, cv=5, verbose=0, pre_dispatch='2*n_jobs')
print('_' * 80)
## print clf
## print
## scores = cross_val_score(clf, X, y, scoring="roc_auc", cv=5)
## print "spectrogram =", scores.mean(), scores.std()

t0 = time()
grid_search.fit(X, y)
print("done in %0.3fs" % (time() - t0))
print

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
