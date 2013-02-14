import numpy as np

from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.preprocessing import normalize

# Loading data
print "Loading data..."
data = np.load("data/train.npz")
X_train = data["X_train"]
y_train = data["y_train"]
data = np.load("data/train_yaafe.npz")
X_train_yaafe = data["X_train_yaafe"]

# Transforming data
print "Transforming data..."
fft = abs(np.fft.fft(X_train))
X_train = np.hstack((normalize(X_train), fft, X_train_yaafe))

# Training
print "Training..."
clf = AdaBoostClassifier(n_estimators=200)
clf.fit(X_train, y_train)

# Predictions
print "Predicting..."
data = np.load("data/test.npz")
X_test = data["X_test"]
data = np.load("data/test_yaafe.npz")
X_test_yaafe = data["X_test_yaafe"]

fft = abs(np.fft.fft(X_test))
X_test = np.hstack((normalize(X_test), fft, X_test_yaafe))

y_proba = clf.predict_proba(X_test)
np.savetxt("adaboost5.txt", y_proba[:, 1])

# Variable importances
np.savetxt("adaboost-var.txt", clf.feature_importances_)

