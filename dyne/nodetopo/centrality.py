"""
Centrality pipes for quantifying node-based topology measurements

Created by: Ankit Khambhati

Change Log
----------
2016/03/06 - Implemented
"""

from __future__ import division
import numpy as np

from base import NodeTopoPipe


class DegrCentral(NodeTopoPipe):
    """
    DegrCentral class for computing weighted degree centrality of the nodes
    """

    def __init__(self):
        self = self

    def _pipe_as_flow(self, signal_packet):
        # Get signal_packet details
        hkey = signal_packet.keys()[0]
        adj = signal_packet[hkey]['data']

        centrality = np.sum(adj, axis=0).reshape(-1, 1)

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


class SyncCentral(NodeTopoPipe):
    """
    SyncCentral class for computing synchronizing/desynchronizing centrality
    of the nodes
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

        centrality = []
        base_sync = global_sync(adj)
        for node_r in np.arange(adj.shape[0]):
            adj_mod = adj.copy()
            adj_mod = np.delete(adj_mod, (node_r), axis=0)
            adj_mod = np.delete(adj_mod, (node_r), axis=1)

            mod_sync = global_sync(adj_mod)
            centrality.append((mod_sync-base_sync) / base_sync)
        centrality = np.array(centrality).reshape(-1, 1)

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
