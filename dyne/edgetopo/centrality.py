"""
Centrality pipes for quantifying edge-based topology measurements

Created by: Ankit Khambhati

Change Log
----------
2016/03/10 - Implemented EdgeSyncCentral
"""

from __future__ import division
import numpy as np

from base import EdgeTopoPipe

class EdgeSyncCentral(EdgeTopoPipe):
    """
    EdgeSyncCentral class for computing synchronizing/desynchronizing centrality
    of the edges
    """

    def __init__(self):
        self = self

    def _pipe_as_flow(self, signal_packet):
        def global_sync(adj_matr):
            """Compute synchronizability"""

            # Get the degree vector of the adj
            deg_vec = np.sum(adj_matr, axis=0)
            deg_matr = np.diag(deg_vec)

            # Laplacian
            lapl = deg_matr - adj_matr

            # Compute eigenvalues and eigenvectors, ensure they are real
            eigval, eigvec = np.linalg.eig(lapl)
            eigval = np.real(eigval)

            # Sort smallest to largest eigenvalue
            eigval = np.sort(eigval)
            sync = np.abs(eigval[1] / eigval[-1]).reshape(1, 1)

            return sync

        # Get signal_packet details
        hkey = signal_packet.keys()[0]
        adj = signal_packet[hkey]['data']
        triu_ix, triu_iy = np.triu_indices_from(adj, k=1)

        centrality = np.zeros_like(adj)
        base_sync = global_sync(adj)
        for n1, n2 in zip(triu_ix, triu_iy):
            adj_mod = adj.copy()
            adj_mod[n1, n2] = 0
            adj_mod[n2, n1] = 0

            mod_sync = global_sync(adj_mod)
            centrality[n1, n2] = (mod_sync-base_sync) / base_sync
            centrality[n2, n1] = (mod_sync-base_sync) / base_sync

        # Dump into signal_packet
        signal_packet[hkey]['data'] = centrality

        return signal_packet
