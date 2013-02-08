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



X, y = load_training_data("data/train.csv", "data/train")
print X, y
