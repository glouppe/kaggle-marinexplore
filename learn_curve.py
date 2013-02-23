import numpy as np

from sklearn.base import clone
from sklearn.utils import check_random_state, shuffle
from sklearn import metrics

from sklearn.externals.joblib import Parallel, delayed


class LearningCurve(object):
    """Creates learning curves by iteratively sampling a subset from
    the given training set, fitting an ``estimator`` on the subset and
    evaluating the estimator on the fixed test set.

    At each iteration the subset size is increased. If
    ``sample_bins`` is an int the subset size is increased by
    ``X_train.shape[0] // sample_bins`` at each iteration until
    ``X_train.shape[0]`` is reached. Otherwise, ``sample_bins`` is expected to
    be an array holding the subset size of each iteration. Each iteration is
    repeated ``num_repetitions`` times using different samples in order
    to create error bars.
    """

    def __init__(self, estimator, sample_bins=10, num_repetitions=10,
                 score_function=metrics.auc_score, random_state=None,
                 n_jobs=1, verbose=0):
        self.base_clf = clone(estimator)
        self.sample_bins = sample_bins
        self.num_repetitions = num_repetitions
        self.score_function = score_function
        self.random_state = check_random_state(random_state)
        self.n_jobs = n_jobs
        self.verbose = verbose

    def run(self, X_train, y_train, X_test, y_test):
        assert X_train.shape[0] == y_train.shape[0]
        if isinstance(self.sample_bins, int):
            sample_bins = np.linspace(0, X_train.shape[0],
                                      self.sample_bins + 1).astype(np.int)
            sample_bins = sample_bins[1:]
        else:
            sample_bins = np.asarray(self.sample_bins, dtype=np.int)

        train_scores = np.zeros((len(sample_bins), self.num_repetitions))
        test_scores = np.zeros((len(sample_bins), self.num_repetitions))

        self.sample_bins = sample_bins
        if self.verbose > 0:
            print("sample_bins: %s" % str(self.sample_bins))

        tasks = (delayed(lc_fit)(i, j, n_samples, X_train, y_train,
                                 X_test, y_test, self.base_clf,
                                 self.score_function)
                 for i, n_samples in enumerate(sample_bins)
                 for j in range(self.num_repetitions))

        out = Parallel(n_jobs=self.n_jobs, pre_dispatch='2*n_jobs',
                       verbose=self.verbose)(tasks)

        # out is a list of 5-tuples (i, n_samples, train_score, test_score)
        for i, j, n_samples, train_score, test_score in out:
            train_scores[i, j] = train_score
            test_scores[i, j] = test_score

        self.train_scores = train_scores
        self.test_scores = test_scores
        print("lc.run() fin")

    def plot(self, title=None, y_label=None, ylim=None):
        """Plot the ``train_scores`` and ``test_scores`` using
        error bars.
        """
        import pylab as pl
        pl.errorbar(self.sample_bins, self.test_scores.mean(axis=1),
                    yerr=self.test_scores.std(axis=1), fmt='r.-',
                    label='test')
        pl.errorbar(self.sample_bins, self.train_scores.mean(axis=1),
                    yerr=self.train_scores.std(axis=1), fmt='b.-',
                    label='train')
        pl.legend(loc='upper right')
        if y_label:
            pl.ylabel(y_label)
        if title:
            pl.title(title)
        if ylim:
            pl.ylim(ylim)


def lc_fit(i, j, n_samples, X_train, y_train, X_test, y_test, clf,
           score_func):
    n = X_train.shape[0]
    print('n_samples: %d' % n_samples)
    idx = shuffle(np.arange(n))[:n_samples]
    X_train = X_train[idx]
    y_train = y_train[idx]
    if len(np.unique(y_train)) != 2:
        raise ValueError('Split has just one class')
    clf.fit(X_train, y_train)

    # FIXME use either predict or decision_function based on score_func
    pred_func = clf.predict
    if hasattr(clf, 'predict_proba'):
        pred_func = lambda X: clf.predict_proba(X)[:, 1].ravel()
    elif hasattr(clf, 'decision_function'):
        pred_func = clf.decision_function

    train_score = score_func(y_train, pred_func(X_train))
    test_score = score_func(y_test, pred_func(X_test))
    return (i, j, n_samples, train_score, test_score)


if __name__ == '__main__':
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.pipeline import FeatureUnion
    from sklearn.cross_validation import train_test_split

    from transform import FlattenTransformer
    from transform import SpectrogramTransformer
    from transform import SpectrogramStatsTransformer

    data = np.load("data/train.npz")

    X = data["X_train"]
    y = data["y_train"]
    #X = X.astype(np.float64)

    st = SpectrogramTransformer(NFFT=256, clip=500, noverlap=0.6, dtype=np.float64,
                                whiten=None, log=True, flatten=False)
    X = st.fit_transform(X)

    tf = FeatureUnion([
        ('spec', FlattenTransformer(scale=1.0)),
        ('sst1', SpectrogramStatsTransformer(axis=1)),
        ('sst0', SpectrogramStatsTransformer(axis=0)),
    ])
    X = tf.fit_transform(X)

    ## clf = LinearSVC(C=0.00001, loss='l1', dual=True, class_weight='auto')
    ## clf = GradientBoostingClassifier(learning_rate=0.1,
    ##                                  max_depth=3, max_features=None,
    ##                                  min_samples_leaf=2,
    ##                                  n_estimators=250, random_state=13)
    clf = ExtraTreesClassifier(n_estimators=50, min_samples_split=3,
                               max_features=None)

    def inv_auc_score(y_true, y_scores):
        return 1.0 - metrics.auc_score(y_true, y_scores)

    lc = LearningCurve(clf, sample_bins=5, num_repetitions=3,
                       score_function=inv_auc_score,
                       random_state=13, n_jobs=3, verbose=11)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                        random_state=42)
    print('_' * 80)
    print('Create learning curve')
    lc.run(X_train, y_train, X_test, y_test)
    lc.plot(title='ExtraTrees(100, 3, None)', y_label='1 - AUC', ylim=(0.0, 0.08))
    import IPython
    IPython.embed()
