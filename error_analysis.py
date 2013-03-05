import numpy as np

import pprint
import hashlib

from datetime import datetime

import matplotlib
matplotlib.use('pdf')

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


def _plot_errors(X, y, y_scores, pdf, spec_func=None, type='fp', k=10):
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

    n_cols = 2
    n_rows = int(np.ceil(k / n_cols))
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                             figsize=(8.27, 11.69),
                             sharey=True, sharex=True)
    axes_iter = axes.flat

    fig.text(0.5, .95, "type: %s | type_label: %d" % (type.upper(), type_label),
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
            ax.set_title("Ind %d - label: %d - score: %.2f" %
                         (i, y[i], y_scores[i]),
                         axes=ax, size=8)
        else:
            n_other_label += 1

    plt.savefig(pdf, format='pdf')
    plt.close()


def error_report(clf, X, y, spec_func=None):
    if hasattr(clf, 'decision_function'):
        y_scores = clf.decision_function(X)
    else:
        y_scores = clf.predict_proba(X)[:, 1]

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
    _plot_errors(X, y, y_scores, pdf, spec_func=None, type='fp', k=10)
    plt.savefig(pdf, format='pdf')
    plt.close()

    pdf.close()

    plt.interactive(True)
