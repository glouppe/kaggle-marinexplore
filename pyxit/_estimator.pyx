# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

cimport cython

import numpy as np
cimport numpy as np

from sklearn.tree._tree cimport Tree

import os
import cPickle


def leaf_transform(trees, np.ndarray[np.float32_t, ndim=2] _X, int n_samples, int n_subwindows):
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int offset_tree = 0
    cdef unsigned int index
    cdef np.float64_t inc = 1.0 / n_subwindows

    cdef Tree tree

    cdef int node_id

    row = []
    col = []
    data = []

    for tree in trees:
        index = 0

        for i in range(n_samples):
            for j in range(n_subwindows):
                node_id = 0

                while tree.children_left[node_id] != -1: # and tree.children_right[node_id, 1] != -1:
                    if _X[index, tree.feature[node_id]] <= tree.threshold[node_id]:
                        node_id = tree.children_left[node_id]
                    else:
                        node_id = tree.children_right[node_id]

                row.append(i)
                col.append(offset_tree + node_id)
                data.append(inc)

                index += 1

        offset_tree += tree.node_count

    return row, col, data, offset_tree


ctypedef np.float64_t DOUBLE

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def inplace_csr_column_scale_max(X, np.ndarray[DOUBLE, ndim=1] maxs=None):
    cdef unsigned int n_samples = X.shape[0]
    cdef unsigned int n_features = X.shape[1]

    cdef np.ndarray[DOUBLE, ndim=1] X_data = X.data
    cdef np.ndarray[int, ndim=1] X_indices = X.indices
    cdef np.ndarray[int, ndim=1] X_indptr = X.indptr

    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int ind

    if maxs is None:
        maxs = np.zeros(n_features, dtype=X.dtype)

        for i in xrange(n_samples):
            for j in xrange(X_indptr[i], X_indptr[i + 1]):
                ind = X_indices[j]

                if X_data[j] > maxs[ind]:
                    maxs[ind] = X_data[j]

    for i in xrange(n_samples):
        for j in xrange(X_indptr[i], X_indptr[i + 1]):
            ind = X_indices[j]

            if maxs[ind] > 0.0:
                X_data[j] /= maxs[ind]

    return X, maxs
