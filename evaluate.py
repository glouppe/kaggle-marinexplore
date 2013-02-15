import numpy as np

from sklearn.cross_validation import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from transform import SpectrogramTransformer

data = np.load("data/train.npz")

X = data["X_train"]
y = data["y_train"]


#clf = LinearSVC(C=10.)

clf = Pipeline(steps=[('spectrogram', SpectrogramTransformer()),
                      ('whitening', PCA(n_components=0.9, whiten=True)),
                      ('sgd', SGDClassifier(n_iter=100))])


print('_' * 80)
print clf
print
scores = cross_val_score(clf, X, y, scoring="roc_auc", cv=3)
print "spectrogram(whitened) =", scores.mean(), scores.std()
