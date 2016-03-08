"""
Classes for Correlation-based Association
"""


from __future__ import division
from ...common.pipe import AdjacencyPipe
import mcompare
import numpy as np


class XCorr(AdjacencyPipe):
    """
    XCorr class for cross-correlation association between sensors in incoming
    signal

    Association matrix between nodes using delayed Pearson correlation to
    determine associations.

    Parameters
    ----------
    alpha: float (default: 0.05)
        Rejection rate to control false positives with surrogate testing
    """

    def __init__(self, pipe_name=None, alpha=0.05, cache=None):
        super(XCorr, self).__init__(pipe_name, cache)
        assert type(alpha) == float
        self.alpha = alpha

    def _func_def(self, signal_packet):
        """
        Compute Cross-Correlation of multivariate signal

        Parameters
        ----------
        signal_packet: dict
            SEE ADJACENCY

        Returns
        -------
        signal_packet: dict
            SEE ADJACENCY
        """
        # Here is the core function
        def xc(signal):
            n_samples_win = signal.shape[0]
            n_nodes = signal.shape[1]
            triuIdx = np.triu_indices(n_nodes, k=1)
            n_edges = triuIdx[0].shape[0]

            # Normalize data
            signal -= signal.mean(axis=0)
            signal /= signal.std(axis=0)

            # Initialize association matrix
            assoc = np.zeros((n_nodes, n_nodes))

            signal_fft = np.fft.rfft(
                np.vstack((signal, np.zeros_like(signal))), axis=0)
            # Iterate over all edges
            for eidx in xrange(n_edges):
                n1 = triuIdx[0][eidx]
                n2 = triuIdx[1][eidx]

                # Normalized cross-correlation
                xcorr = 1 / n_samples_win * np.fft.irfft(
                    signal_fft[:, n1] * np.conj(signal_fft[:, n2]))
                assoc[n1, n2] = np.max(np.abs(xcorr))

            assoc += assoc.T

            return assoc

        # Apply to the real and get the null
        real_assoc = xc(signal_packet['win'])

        # If available use surrogate data, compute adjacency
        if 'surr_win' in signal_packet.keys():
            surr_win = signal_packet['surr_win']
            surr_assoc = np.zeros((surr_win.shape[0],
                                  surr_win.shape[2],
                                  surr_win.shape[2]))
            for s_idx, s_win in enumerate(surr_win):
                surr_assoc[s_idx, :, :] = xc(s_win)

            # Get null merged distribution of associations
            triu_idx = np.triu_indices(surr_win.shape[2], k=1)
            null_assoc = surr_assoc[:, triu_idx[0], triu_idx[1]].reshape(-1)

            # Compute p-value
            pval_assoc = np.zeros_like(real_assoc)
            for idx in xrange(len(triu_idx[0])):
                assoc_val = real_assoc[triu_idx[0][idx], triu_idx[1][idx]]
                n_sig_h0 = len(np.flatnonzero(null_assoc > assoc_val))
                pval_assoc[triu_idx[0][idx], triu_idx[1][idx]] = \
                    n_sig_h0 / len(null_assoc)
            pval_assoc += pval_assoc.T

            # Compute FDR
            adj = mcompare.FDR(real_assoc, pval_assoc, alpha=self.alpha)
        else:
            adj = real_assoc

        return adj


class Corr(AdjacencyPipe):
    """
    Corr class for Pearson correlation association between sensors in incoming
    signal

    Association matrix between nodes using Pearson correlation to
    determine associations.

    Parameters
    ----------
    alpha: float (default: 0.05)
        Rejection rate to control false positives with surrogate testing
    """

    def __init__(self, pipe_name=None, alpha=0.05, cache=None):
        super(Corr, self).__init__(pipe_name, cache)
        assert type(alpha) == float
        self.alpha = alpha

    def _func_def(self, signal_packet):
        """
        Compute Pearson Correlation of multivariate signal

        Parameters
        ----------
        signal_packet: dict
            SEE ADJACENCY

        Returns
        -------
        signal_packet: dict
            SEE ADJACENCY
       """
        def mag_corr(signal):
            assoc = np.abs(np.corrcoef(signal.T))

            return assoc

        # Apply to the real and get the null
        real_assoc = mag_corr(signal_packet['win'])

        # If available use surrogate data, compute adjacency
        if 'surr_win' in signal_packet.keys():
            surr_win = signal_packet['surr_win']
            surr_assoc = np.zeros((surr_win.shape[0],
                                  surr_win.shape[2],
                                  surr_win.shape[2]))
            for s_idx, s_win in enumerate(surr_win):
                surr_assoc[s_idx, :, :] = mag_corr(s_win)

            # Get null merged distribution of associations
            triu_idx = np.triu_indices(surr_win.shape[2], k=1)
            null_assoc = surr_assoc[:, triu_idx[0], triu_idx[1]].reshape(-1)

            # Compute p-value
            pval_assoc = np.zeros_like(real_assoc)
            for idx in xrange(len(triu_idx[0])):
                assoc_val = real_assoc[triu_idx[0][idx], triu_idx[1][idx]]
                n_sig_h0 = len(np.flatnonzero(null_assoc > assoc_val))
                pval_assoc[triu_idx[0][idx], triu_idx[1][idx]] = \
                    n_sig_h0 / len(null_assoc)
            pval_assoc += pval_assoc.T

            # Compute FDR
            adj = mcompare.FDR(real_assoc, pval_assoc, alpha=self.alpha)
        else:
            adj = real_assoc

        return adj
