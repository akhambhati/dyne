"""
Classes for Coherence-based Association
"""

from __future__ import division
from ...common.pipe import AdjacencyPipe
import mcompare
import numpy as np
from mtspec import mt_coherence, mtspec
from scipy.signal import coherence
import matplotlib.pyplot as plt


class Coh(AdjacencyPipe):
    """
    Coh class for spectral coherence estimation between sensors in incoming
    signal

    Association matrix between nodes using Welch's method for spectral
    estimation determine associations.

    Parameters
    ----------
    window: str
        Desired window to use. See Scipy get_window for a list of windows.
    secperseg: float
        Length of each segment in seconds. Recommended half of window length.
    pctoverlap: float (0<x<1)
        Percent overlap between segments. Recommended values of 50 pct.
    cf: tuple
        Frequency range over which to compute coherence [-NW+C, C+NW]
    alpha: float (default: 0.05)
        Rejection rate to control false positives with surrogate testing
    """

    def __init__(self, pipe_name=None, window=None, secperseg=None,
                 pctoverlap=None, cf=None, alpha=0.05, cache=None):
        super(Coh, self).__init__(pipe_name, cache)

        assert type(window) == str
        self.window = window

        assert type(secperseg) == float
        self.secperseg = secperseg

        assert type(pctoverlap) == float
        assert (pctoverlap > 0) and (pctoverlap < 1)
        self.pctoverlap = pctoverlap

        assert type(cf) == tuple or type(cf) == list
        self.cf = cf

        assert type(alpha) == float
        self.alpha = alpha

    def _func_def(self, signal_packet):
        """
        Compute spectral coherence

        Parameters
        ----------
        signal_packet: dict
            SEE ADJACENCY

        Returns
        -------
        signal_packet: dict
            SEE ADJACENCY
        """
        def coh(signal, fs):
            n_sample_win = signal.shape[0]
            n_node = signal.shape[1]

            # Calculate connection sizes
            triu_idx = np.triu_indices(n_node, k=1)
            n_connect = int(0.5*n_node*(n_node-1))

            # Initialize association matrix
            assoc = np.zeros((n_node, n_node))

            # Test signal integrity
            signal_stdev = np.std(signal, axis=0)
            low_amp_var_idx = np.flatnonzero(signal_stdev < 1e-4)
            if len(low_amp_var_idx) > 0:
                return assoc

            # Derive signal segmenting for coherence estimation
            nperseg = int(self.secperseg*fs)
            noverlap = int(self.secperseg*fs*self.pctoverlap)

            # Compute all coherences
            freq, Cxy = coherence(signal[:, triu_idx[0]],
                                  signal[:, triu_idx[1]],
                                  fs=fs, window=self.window,
                                  nperseg=nperseg, noverlap=noverlap,
                                  axis=0)

            # Find closest frequency to the desired center frequency
            cf_idx = np.flatnonzero((freq >= self.cf[0]) &
                                    (freq <= self.cf[1]))

            # Store coherence in association matrix
            assoc[triu_idx[0], triu_idx[1]] = np.mean(Cxy[cf_idx, :], axis=0)

            assoc += assoc.T

            return assoc

        signal = signal_packet['win']
        fs = signal_packet['fs']

        real_assoc = coh(signal, fs)

        # If available use surrogate data, compute adjacency
        if 'surr_win' in signal_packet.keys():
            surr_win = signal_packet['surr_win']
            surr_assoc = np.zeros((surr_win.shape[0],
                                  surr_win.shape[2],
                                  surr_win.shape[2]))
            for s_idx, s_win in enumerate(surr_win):
                surr_assoc[s_idx, :, :] = coh(s_win, fs)

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

class MTCoh(AdjacencyPipe):
    """
    MTCoh class for multitaper coherence between sensors in incoming
    signal

    Association matrix between nodes using Pearson correlation to
    determine associations.

    Parameters
    ----------
    NW: float
        The time half bandwidth resolution of the estimate [-NW, NW];
        such that resolution is 2*NW
    k: int
        Number of Slepian sequences to use (Usually < 2*NW-1)
    cf: tuple
        Frequency range over which to compute coherence [-NW+C, C+NW]
    alpha: float (default: 0.05)
        Rejection rate to control false positives with surrogate testing
    """

    def __init__(self, pipe_name=None, NW=None, k=None, cf=None, alpha=0.05,
                 cache=None):
        super(MTCoh, self).__init__(pipe_name, cache)
        assert type(NW) == float
        self.NW = NW

        assert type(k) == int
        self.k = k

        assert type(cf) == tuple or type(cf) == list
        self.cf = cf

        assert type(alpha) == float
        self.alpha = alpha

    def _func_def(self, signal_packet):
        """
        Compute multitaper coherence

        Parameters
        ----------
        signal_packet: dict
            SEE ADJACENCY

        Returns
        -------
        signal_packet: dict
            SEE ADJACENCY
        """
        def coh(signal, fs):
            n_sample_win = signal.shape[0]
            n_node = signal.shape[1]

            # Calculate connection sizes
            triu_idx = np.triu_indices(n_node, k=1)
            n_connect = int(0.5*n_node*(n_node-1))

            # Initialize association matrix
            assoc = np.zeros((n_node, n_node))

            # Test signal integrity
            signal_stdev = np.std(signal, axis=0)
            low_amp_var_idx = np.flatnonzero(signal_stdev < 1e-4)
            if len(low_amp_var_idx) > 0:
                return assoc

            # Compute all coherences
            for idx_cnct in xrange(n_connect):
                ix = triu_idx[0][idx_cnct]
                iy = triu_idx[1][idx_cnct]

                out = mt_coherence(1.0/fs, signal[:, ix], signal[:, iy],
                                   self.NW, self.k,
                                   int(n_sample_win/2), 0.95, iadapt=1,
                                   cohe=True, freq=True)

                # Find closest frequency to the desired center frequency
                #cf_idx = np.argmin(np.abs(out['freq'] - self.cf))
                cf_idx = np.flatnonzero((out['freq'] >= self.cf[0]) &
                                        (out['freq'] <= self.cf[1]))

                # Store coherence in association matrix
                assoc[ix, iy] = np.mean(out['cohe'][cf_idx])


            assoc += assoc.T

            return assoc

        signal = signal_packet['win']
        fs = signal_packet['fs']

        real_assoc = coh(signal, fs)

        # If available use surrogate data, compute adjacency
        if 'surr_win' in signal_packet.keys():
            surr_win = signal_packet['surr_win']
            surr_assoc = np.zeros((surr_win.shape[0],
                                  surr_win.shape[2],
                                  surr_win.shape[2]))
            for s_idx, s_win in enumerate(surr_win):
                surr_assoc[s_idx, :, :] = coh(s_win, fs)

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
