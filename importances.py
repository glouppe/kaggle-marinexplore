import numpy as np
import pylab as pl

imp = np.loadtxt("feature-importances-rf2.txt")

datasets = [
    ("ceps_2000", 198, 9),
    ("specs_2000", 198, 9),
    ("ceps_4000", 98, 9),
    ("specs_4000", 98, 13),
    ("ceps_8000", 48, 9),
    ("specs_8000", 48, 17),
    ("ceps_16000", 23, 9),
    ("specs_16000", 23, 21),
    ("ceps_32000", 11, 9),
    ("specs_32000", 11, 25),
    ("mfcc_8000", 48, 13),
    ("mfcc_16000", 23, 13),
    ("mfcc_32000", 11, 13),
    ("mfcc_64000", 4, 13),
    ("wiener1spectro", 65, 30),
]

features = []

for name, d0, d1 in datasets:
    features.append((name, d0*d1))
    features.append(("%s_stats (axis=1)" % name, 6*d0))
    features.append(("%s_stats (axis=0)" % name, 6*d1))


indices = {}
start = 0

for name, count in features:
    indices[name] = (start, start + count)
    pl.plot(range(start, start+count), imp[start:start+count], label=name)
    start += count

pl.legend(loc="upper left", prop={'size':7})
pl.show()

