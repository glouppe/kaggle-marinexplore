import numpy as np

import pprint
import hashlib

from datetime import datetime

import matplotlib
matplotlib.use('pdf')
matplotlib.rc('xtick', labelsize=6)
matplotlib.rc('ytick', labelsize=6)

from matplotlib.backends.backend_pdf import PdfPages
import pylab as plt


from sklearn.metrics import roc_curve, auc


def _plot_roc(y, y_scores, ax):
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    #ax.clf()
    ax.plot(fpr, tpr, label='AUC = %0.4f' % roc_auc)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")


def _plot_errors(X, ind, y, y_scores, pdf, spec_func=None, type='fp', k=10):
    ranking = y_scores.argsort()
    if type[-1] == 'p':
        print('sorting scores in desc order')
        ranking = ranking[::-1]

    def swap(label):
        return 1 if label == 0 else 0

    type_label = 1
    if type[-1] == 'n':
        type_label = 0

    if type[0] == 'f':
        # swap labels
        type_label = swap(type_label)

    if spec_func is None:
        spec_func = lambda x, ax: ax.specgram(x, NFFT=256, Fs=2000)

    n_cols = 3
    n_rows = int(np.ceil(k / n_cols))
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                             figsize=(8.27, 11.69),
                             sharey=True, sharex=True)
    axes_iter = axes.flat

    fig.text(0.5, .95, "%s - %d samples" % (type.upper(), y.shape[0]),
             horizontalalignment='center', size=20)

    n_other_label = 0
    for pos, i in enumerate(ranking):
        if y[i] == type_label:
            print('plotting rank %d' % i)
            try:
                ax = next(axes_iter)
            except StopIteration:
                break

            spec_func(X[i], ax)
            # rank - index - label - score
            ax.set_title("r %d - i %d - l %d - s %.2f" %
                         (pos, ind[i], y[i], y_scores[i]),
                         axes=ax, size=6)
        else:
            n_other_label += 1

    plt.savefig(pdf, format='pdf')
    plt.close()


def error_report(clf, X, y, y_scores=None, ind=None, spec_func=None):
    """Generate error report as a multi page pdf.

    This functions plots the ROC curve of ``clf`` and spectrograms
    for the top ``k`` false negatives, false positives, true positives,
    and true negatives.

    Parameters
    ----------
    clf : BaseEstimator
        A trained classifier
    X : ndarray
        A data array, used to generate the spectrograms (using ``spec_func``)
        and optionally ``y_scores``.
    """
    if y_scores is None:
        if hasattr(clf, 'decision_function'):
            y_scores = clf.decision_function(X)
        else:
            y_scores = clf.predict_proba(X)[:, 1]

    if ind is None:
        ind = np.arange(X.shape[0])

    plt.interactive(False)

    signature = hashlib.md5(repr(clf)).hexdigest()
    fname = 'error_report_%s.pdf' % signature
    pdf = PdfPages(fname)

    # frontpage
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.5, .9, "Error Report", horizontalalignment='center',
             size=20)
    fig.text(0.5, .75, str(datetime.now()), horizontalalignment='center',
             size=12)
    fig.text(0.5, .5, pprint.pformat(clf), horizontalalignment='center',
             size=10)
    plt.savefig(pdf, format='pdf')
    plt.close()

    # roc curve
    print('_' * 80)
    print 'roc curve'
    print
    fig = plt.figure(figsize=(8.27, 8.27))
    _plot_roc(y, y_scores, fig.gca())
    plt.savefig(pdf, format='pdf')
    plt.close()

    fig = plt.figure(figsize=(8.27, 8.27))
    _plot_errors(X, ind, y, y_scores, pdf, spec_func=None, type='fp', k=20)
    plt.savefig(pdf, format='pdf')
    plt.close()

    fig = plt.figure(figsize=(8.27, 8.27))
    _plot_errors(X, ind, y, y_scores, pdf, spec_func=None, type='fn', k=20)
    plt.savefig(pdf, format='pdf')
    plt.close()

    fig = plt.figure(figsize=(8.27, 8.27))
    _plot_errors(X, ind, y, y_scores, pdf, spec_func=None, type='tp', k=20)
    plt.savefig(pdf, format='pdf')
    plt.close()

    fig = plt.figure(figsize=(8.27, 8.27))
    _plot_errors(X, ind, y, y_scores, pdf, spec_func=None, type='tn', k=20)
    plt.savefig(pdf, format='pdf')
    plt.close()

    pdf.close()

    plt.interactive(True)


if __name__ == '__main__':
    from sklearn.cross_validation import train_test_split
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import ExtraTreesClassifier

    from sklearn.pipeline import Pipeline

    from transform import SpectrogramTransformer

    from ranking import RankSVM
    from ranking import SVMPerf
    from ranking import RGradientBoostingClassifier

    import IPython

    data = np.load("data/train_small.npz")

    X = data["X_train"]
    y = data["y_train"]

    clf = LinearSVC(C=1e-5, tol=0.001, loss='l1', dual=True)

    clf = Pipeline(steps=[('spectrogram',
                           SpectrogramTransformer(NFFT=256, noverlap=0.5,
                                                  dtype=np.float64)),
                          ('svm', clf)])

    ind = np.arange(X.shape[0])

    X_train, X_test, y_train, y_test, ind_train, ind_test = train_test_split(
        X, y, ind, test_size=0.5, random_state=42)

    clf.fit(X_train, y_train)

    from error_analysis import error_report

    error_report(clf, X_test, y=y_test, ind=ind_test, spec_func=None)
