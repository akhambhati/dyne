"""
Helper Routines for Multiple Comparisons Hypothesis Testing
"""


from __future__ import division
import numpy as np


def FDR(real_assoc, pval_assoc, alpha=0.05):
    """
    FDR class for multiple comparisons false discovery rate (FDR) analysis

    Find network adjacency matrix based on FDR between real and surrogate
    association matrices.

    Parameters
    ----------
    real_assoc: ndarray, shape: [n_node x n_node]
        Real association matrix

    pval_assoc: ndarray, shape: [n_node x n_node]
        P-value association matrix

    alpha: float
        Rejection rate to control false positives
    """

    assert type(alpha) == np.float

    assert type(real_assoc) == np.ndarray
    assert len(real_assoc.shape) == 2
    assert real_assoc.shape[0] == real_assoc.shape[1]

    assert type(pval_assoc) == np.ndarray
    assert len(pval_assoc.shape) == 2
    assert pval_assoc.shape[0] == pval_assoc.shape[1]

    n_nodes = real_assoc.shape[1]

    # Get upper triangle, assume symmetry
    triuIdx = np.triu_indices_from(real_assoc, k=1)
    n_triu = triuIdx[0].shape[0]

    # Flatten the symmetrical arrays (upper triangle)
    real_assoc_flat = real_assoc[triuIdx[0], triuIdx[1]]
    pval_assoc_flat = pval_assoc[triuIdx[0], triuIdx[1]]

    # False discovery rate
    adj_update = np.zeros(n_triu) * np.nan
    pvals_idx = np.argsort(pval_assoc_flat)
    for i, pvidx in enumerate(pvals_idx):
        if pval_assoc_flat[pvidx] <= (i + 1) / n_triu * alpha:
            adj_update[pvidx] = real_assoc_flat[pvidx]
        else:
            break

    # Reformulate array to symmetric adjacency matrix
    adj = np.zeros((n_nodes, n_nodes))
    adj[triuIdx] = adj_update
    adj += adj.T

    return adj
