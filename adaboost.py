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
data = np.load("data/test.npz")
X_test = data["X_test"]

# Transforming data
print "Transforming data..."
X_train = normalize(X_train)
X_test = normalize(X_test)

fft = np.fft.fft(X_train)
X_train = np.hstack((X_train, fft.real, fft.imag))
fft = np.fft.fft(X_test)
X_test = np.hstack((X_test, fft.real, fft.imag))

# Training
print "Training..."
clf = AdaBoostClassifier(n_estimators=200)
clf.fit(X_train, y_train)

#from sklearn.metrics import auc_score
#print "Score =", clf.score(X_train, y_train)
#print "AUC =", auc_score(y_train, clf.predict(X_train))

# Predictions
print "Predicting..."
y_proba = clf.predict_proba(X_test)
np.savetxt("adaboost4.txt", y_proba[:, 1])
print np.mean(y_proba[:, 1])

