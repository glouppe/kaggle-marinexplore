import numpy as np

from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import normalize

# Loading data
print "Loading data..."
data = np.load("data/train.npz")
X_train = data["X_train"]
y_train = data["y_train"]

# Transforming data
print "Transforming data..."
fft = abs(np.fft.fft(X_train))
X_train = np.hstack((normalize(X_train), fft))

# Training
print "Training..."
clf = AdaBoostClassifier(n_estimators=200)
clf.fit(X_train, y_train)

# Predictions
print "Predicting..."
data = np.load("data/test.npz")
X_test = data["X_test"]

fft = abs(np.fft.fft(X_test))
X_test = np.hstack((normalize(X_test), fft))

y_proba = clf.predict_proba(X_test)
np.savetxt("adaboost5.txt", y_proba[:, 1])
