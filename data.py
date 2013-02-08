#! /usr/bin/env python
# -*- coding: utf-8 -*-

import aifc
import numpy as np
import os
import os.path
import struct


def load_aiff(filename):
    # Load data
    fd = aifc.open(filename, "r")
    sample_width = fd.getsampwidth()
    n_frames = fd.getnframes()
    n_channels = fd.getnchannels()
    data = fd.readframes(fd.getnframes())
    fd.close()

    # Convert bytes into Numpy array
    types = {2:'h', 4:'i', 8:'l'}
    fmt = "<%d%s" % (n_frames * n_channels, types[sample_width])
    samples = np.asarray(struct.unpack(fmt, data), np.float32)

    return samples


def load_training_data(file_labels, dir_aiff):
    X = []
    y = []

    fd_labels = open(file_labels, "r")
    for line in fd_labels:
        filename, label = line.strip().split(",")
        X.append(load_aiff(os.path.join(dir_aiff, filename)))
        y.append(int(label))
    fd_labels.close()

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int)


def load_test_data(dir_aiff, n=54503):
    X = []

    for i in xrange(1, n+1):
        filename = "test%d.aiff" % i
        X.append(load_aiff(os.path.join(dir_aiff, filename)))

    return np.array(X, dtype=np.float32)


if __name__ == "__main__":
    X_train, y_train = load_training_data("data/train.csv", "data/train")
    X_test = load_test_data("data/test")

    # Save for later as numpy arrays
    fd = open("data/train.npz", "wb")
    np.savez(fd, X_train=X_train, y_train=y_train)
    fd.close()

    fd = open("data/test.npz", "wb")
    np.savez(fd, X_test=X_test)
    fd.close()

