"""
Classes for centrality measures on network edges
"""

from __future__ import division
from ...common.pipe import EdgemetricPipe
import numpy as np


class DriveSync(EdgemetricPipe):
    """
    DriveSync class for edge roles in global network synchronizability
    """

    def __init__(self, pipe_name=None, cache=None):
        super(DriveSync, self).__init__(pipe_name, cache)

    def _func_def(self, signal_packet):
        """
        Compute driven synchronizability

        Parameters
        ----------
        signal_packet: dict
            SEE EGDEMETRIC

        Returns
        -------
        centrality: ndarray, shape: [1 x n_edge]
            Vector of driven synchronizability for each edge
        """

        def lapl_sync(adj):
            """
            Compute synchronizability
            """

            # Get the degree vector of the adj
            deg_vec = np.sum(adj, axis=0)
            deg_matr = np.diag(deg_vec)

            # Laplacian
            lapl = deg_matr - adj

            # Compute eigenvalues and eigenvectors, ensure they are real
            eigval, eigvec = np.linalg.eig(lapl)
            eigval = np.real(eigval)

            # Sort smallest to largest eigenvalue
            eigval = np.sort(eigval)
            sync = np.abs(eigval[1] / eigval[-1]).reshape(1, 1)

            return sync

        adj = signal_packet['adj']

        # All non-significant connections are zero
        adj[np.isnan(adj)] = 0
        triu_idx = np.triu_indices_from(adj, k=1)

        centrality = []
        base_sync = lapl_sync(adj)
        for n_ix, n_iy in zip(triu_idx[0], triu_idx[1]):
            adj_mod = adj.copy()
            adj_mod[n_ix, n_iy] = 0
            adj_mod[n_iy, n_ix] = 0

            mod_sync = lapl_sync(adj_mod)
            centrality.append((mod_sync-base_sync) / base_sync)

        centrality = np.array(centrality)
        return centrality
