"""
Classes for Coherence-based Association
"""

from __future__ import division
from ...common.pipe import NodemetricPipe
import numpy as np


class Eigenvector(NodemetricPipe):
    """
    Eigenvector class for eigenvector centrality of the nodes in incoming
    signal
    """

    def __init__(self, pipe_name=None, cache=None):
        super(Eigenvector, self).__init__(pipe_name, cache)

    def _func_def(self, signal_packet):
        """
        Compute eigenvector centrality

        Parameters
        ----------
        signal_packet: dict
            SEE NODEMETRIC

        Returns
        -------
        centrality: ndarray, shape: [1 x n_node]
            Vector of eigenvector centrality for each node
        """
        adj = signal_packet['adj']

        # All non-significant connections are zero
        adj[np.isnan(adj)] = 0

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

        return centrality


class LaplEigenvector(NodemetricPipe):
    """
    LaplEigenvector class for eigenvector centrality of the nodes based on the
    Laplacian matrix of the incoming signal
    """

    def __init__(self, pipe_name=None, cache=None):
        super(LaplEigenvector, self).__init__(pipe_name, cache)

    def _func_def(self, signal_packet):
        """
        Compute eigenvector centrality

        Parameters
        ----------
        signal_packet: dict
            SEE NODEMETRIC

        Returns
        -------
        centrality: ndarray, shape: [1 x n_node]
            Vector of eigenvector centrality for each node
        """
        adj = signal_packet['adj']

        # All non-significant connections are zero
        adj[np.isnan(adj)] = 0
        I = np.diag(np.ones(adj.shape[0]))

        # Get the degree vector of the adj
        deg_vec = np.sum(adj, axis=0)
        deg_matr = np.diag(deg_vec**(-0.5))

        # Normalized Laplacian
        lapl = I - np.dot(np.dot(deg_matr, adj), deg_matr)

        # Compute eigenvalues and eigenvectors, ensure they are real
        eigval, eigvec = np.linalg.eig(lapl)
        eigval = np.real(eigval)
        eigvec = np.real(eigvec)

        # Sort largest to smallest eigenvalue
        sorted_idx = np.argsort(eigval)
        centrality = np.abs(eigvec[:, sorted_idx[-1]])

        return centrality


class Synchronizability(NodemetricPipe):
    """
    Synchronizability class for global network synchronizability
    """

    def __init__(self, pipe_name=None, cache=None):
        super(Synchronizability, self).__init__(pipe_name, cache)

    def _func_def(self, signal_packet):
        """
        Compute synchronizability

        Parameters
        ----------
        signal_packet: dict
            SEE NODEMETRIC

        Returns
        -------
        centrality: ndarray, shape: [1 x n_node]
            Vector of synchronizability for each node
        """
        adj = signal_packet['adj']

        # All non-significant connections are zero
        adj[np.isnan(adj)] = 0

        # Get the degree vector of the adj
        deg_vec = np.sum(adj, axis=0)
        deg_matr = np.diag(deg_vec)

        # Laplacian
        lapl = deg_matr - adj

        # Compute eigenvalues and eigenvectors, ensure they are real
        eigval, eigvec = np.linalg.eig(lapl)
        eigval = np.real(eigval)
        eigvec = np.real(eigvec)

        # Sort largest to smallest eigenvalue
        eigval = np.sort(eigval)
        centrality = np.abs(eigval[1] / eigval[-1]).reshape(1, 1)

        return centrality


class Laplacian(NodemetricPipe):
    """
    Laplacian class for laplacian centrality of the nodes based on the
    change in laplacian energy as nodes are removed from the network
    """

    def __init__(self, pipe_name=None, cache=None):
        super(Laplacian, self).__init__(pipe_name, cache)

    def _func_def(self, signal_packet):
        """
        Compute eigenvector centrality

        Parameters
        ----------
        signal_packet: dict
            SEE NODEMETRIC

        Returns
        -------
        centrality: ndarray, shape: [1 x n_node]
            Vector of eigenvector centrality for each node
        """

        def lapl_energy(adj):
            """
            Compute laplacian energy
            """

            # Get degree
            degr = np.sum(adj, axis=0)

            # Get weights
            triu_idx = np.triu_indices_from(adj, k=1)
            weight = adj[triu_idx[0], triu_idx[1]]

            energy = np.sum(degr**2) + 2*np.sum(weight**2)
            return energy

        adj = signal_packet['adj']

        # All non-significant connections are zero
        adj[np.isnan(adj)] = 0

        centrality = []
        base_energy = lapl_energy(adj)
        for node_r in np.arange(adj.shape[0]):
            adj_mod = adj.copy()
            adj_mod = np.delete(adj_mod, (node_r), axis=0)
            adj_mod = np.delete(adj_mod, (node_r), axis=1)

            mod_energy = lapl_energy(adj_mod)
            centrality.append((base_energy-mod_energy) / base_energy)

        centrality = np.array(centrality)
        return centrality


class DriveSync(NodemetricPipe):
    """
    DriveSync class for node roles in global network synchronizability
    """

    def __init__(self, pipe_name=None, cache=None):
        super(DriveSync, self).__init__(pipe_name, cache)

    def _func_def(self, signal_packet):
        """
        Compute driven synchronizability

        Parameters
        ----------
        signal_packet: dict
            SEE NODEMETRIC

        Returns
        -------
        centrality: ndarray, shape: [1 x n_node]
            Vector of driven synchronizability for each node
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

        centrality = []
        base_sync = lapl_sync(adj)
        for node_r in np.arange(adj.shape[0]):
            adj_mod = adj.copy()
            adj_mod = np.delete(adj_mod, (node_r), axis=0)
            adj_mod = np.delete(adj_mod, (node_r), axis=1)

            mod_sync = lapl_sync(adj_mod)
            centrality.append((mod_sync-base_sync) / base_sync)

        centrality = np.array(centrality)
        return centrality
