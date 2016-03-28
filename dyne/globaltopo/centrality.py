"""
Centrality pipes for quantifying node-based topology measurements

Created by: Ankit Khambhati

Change Log
----------
2016/03/10 - Implemented DegrCentral, EvecCentral, SyncCentral pipes
"""

from __future__ import division
import numpy as np

from ..base import GlobalTopoPipe


class Synchronizability(GlobalTopoPipe):
    """
    Synchronizability class for computing synchronizability of the network
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

        base_sync = global_sync(adj)

        # Dump into signal_packet
        new_packet = {}
        new_packet[hkey] = {
            'data': np.array(base_sync).reshape(1, 1),
            'meta': {
                'time': signal_packet[hkey]['meta']['time']
            }
        }

        return new_packet
