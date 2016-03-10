"""
Centrality pipes for quantifying node-based topology measurements

Created by: Ankit Khambhati

Change Log
----------
2016/03/06 - Implemented
"""

from __future__ import division
import numpy as np

from errors import check_type
from base import NodeTopoPipe


class EvecCentral(NodeTopoPipe):
    """
    EvecCentral class for computing eigenvector centrality of the nodes
    """

    def __init__(self):
        self = self

    def _pipe_as_flow(self, signal_packet):
        # Get signal_packet details
        hkey = signal_packet.keys()[0]
        adj = signal_packet[hkey]['data']

        # Add 1s along the diagonal to make positive definite
        adj[np.diag_indices_from(adj)] = 1

        # Compute eigenvalues and eigenvectors, ensure they are real
        eigval, eigvec = np.linalg.eig(adj)
        eigval = np.real(eigval)
        eigvec = np.real(eigvec)

        # Sort largest to smallest eigenvalue
        sorted_idx = np.argsort(eigval)[::-1]
        largest_idx = sorted_idx[0]
        centrality = np.abs(eigvec[:, largest_idx])
        centrality = centrality.reshape(-1, 1)

        # Dump into signal_packet
        new_packet = {}
        new_packet[hkey] = {
            'data': centrality,
            'meta': {
                'ax_0': signal_packet[hkey]['meta']['ax_0'],
                'time': signal_packet[hkey]['meta']['time']
            }
        }

        return new_packet
