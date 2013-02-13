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
X_train = np.hstack((X_train, fft, X_train_yaafe))

# Training
print "Training..."
clf = ExtraTreesClassifier(n_estimators=500, min_samples_split=10, max_features=1000, n_jobs=4)
params = {"max_features": [500, 1000, 2000],
          "min_samples_split": [10, 50, 100]}

grid = GridSearchCV(clf, params, cv=ShuffleSplit(n=len(X_train), test_size=0.666), scoring="roc_auc", verbose=3)
grid.fit(X_train, y_train)

print grid.grid_scores_
print grid.best_estimator_

# # Predictions
# del X_train
# del X_train_yaafe

# print "Predicting..."
# data = np.load("data/test.npz")
# X_test = data["X_test"]
# data = np.load("data/test_yaafe.npz")
# X_test_yaafe = data["X_test_yaafe"]

# X_test = normalize(X_test)
# fft = abs(np.fft.fft(X_test))
# X_test = np.hstack((X_test, X_test_yaafe, fft))

# y_proba = clf.predict_proba(X_test)
# np.savetxt("et3.txt", y_proba[:, 1])

# # Variable importances
# np.savetxt("et-var.txt", clf.feature_importances_)
