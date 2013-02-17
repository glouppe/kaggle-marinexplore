import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import normalize
from features import spectrogram, spectrogram_stats

# Loading data
print "Loading data..."
data = np.load("data/train.npz")
X_train = data["X_train"]
y_train = data["y_train"]

# Transforming data
print "Transforming data..."
X_train = np.hstack((spectrogram(X_train, upper=500), spectrogram_stats(X_train, upper=500)))

# Training
print "Training..."
clf = ExtraTreesClassifier(n_estimators=500, min_samples_split=10, max_features=None, n_jobs=2)
clf.fit(X_train, y_train)

# Predictions
print "Predicting..."
data = np.load("data/test.npz")
X_test = data["X_test"]
X_test = np.hstack((spectrogram(X_test, upper=500), spectrogram_stats(X_test, upper=500)))

y_proba = clf.predict_proba(X_test)
np.savetxt("et4.txt", y_proba[:, 1])
