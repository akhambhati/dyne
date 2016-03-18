"""
Coherence pipes for quantifying signal similarity (i.e. connectivity)

Created by: Ankit Khambhati

Change Log
----------
2016/03/06 - Implemented WelchCoh and MTCoh pipes
"""

from __future__ import division
import numpy as np
from mtspec import mt_coherence, mtspec
from scipy.signal import coherence
import matplotlib.pyplot as plt

from ..errors import check_type
from ..base import AdjacencyPipe


class WelchCoh(AdjacencyPipe):
    """
    WelchCoh pipe for spectral coherence estimation using Welch's method

    Parameters
    ----------
        window: str
            Desired window to use. See Scipy get_window for a list of windows.

        secperseg: float
            Length of each segment in seconds. Recommended half of window length.

        pctoverlap: float (0<x<1)
            Percent overlap between segments. Recommended values of 50 pct.

        cf: list
            Frequency range over which to compute coherence [-NW+C, C+NW]
    """

    def __init__(self, window, secperseg, pctoverlap, cf):
        # Standard param checks
        check_type(window, str)
        check_type(secperseg, float)
        check_type(pctoverlap, float)
        check_type(cf, list)
        if not len(cf) == 2:
            raise Exception('Must give a frequency range in list of length 2')
        if (pctoverlap > 1) or (pctoverlap < 0):
            raise Exception('Percent overlap must be a positive fraction')

        # Assign to instance
        self.window = window
        self.secperseg = secperseg
        self.pctoverlap = pctoverlap
        self.cf = cf

    def _pipe_as_flow(self, signal_packet):
        # Get signal_packet details
        hkey = signal_packet.keys()[0]
        ax_0_ix = signal_packet[hkey]['meta']['ax_0']['index']
        ax_1_ix = signal_packet[hkey]['meta']['ax_1']['index']
        signal = signal_packet[hkey]['data']
        fs = np.int(np.mean(1./np.diff(ax_0_ix)))

        # Assume undirected connectivity
        triu_ix, triu_iy = np.triu_indices(len(ax_1_ix), k=1)

        # Initialize association matrix
        adj = np.zeros((len(ax_1_ix), len(ax_1_ix)))

        # Derive signal segmenting for coherence estimation
        nperseg = int(self.secperseg*fs)
        noverlap = int(self.secperseg*fs*self.pctoverlap)

        freq, Cxy = coherence(signal[:, triu_ix],
                              signal[:, triu_iy],
                              fs=fs, window=self.window,
                              nperseg=nperseg, noverlap=noverlap,
                              axis=0)

        # Find closest frequency to the desired center frequency
        cf_idx = np.flatnonzero((freq >= self.cf[0]) &
                                (freq <= self.cf[1]))

        # Store coherence in association matrix
        adj[triu_ix, triu_iy] = np.mean(Cxy[cf_idx, :], axis=0)
        adj += adj.T

        new_packet = {}
        new_packet[hkey] = {
            'data': adj,
            'meta': {
                'ax_0': signal_packet[hkey]['meta']['ax_1'],
                'ax_1': signal_packet[hkey]['meta']['ax_1'],
                'time': {
                    'label': 'Time (sec)',
                    'index': np.float(ax_0_ix[-1])
                }
            }
        }

        return new_packet


class MTCoh(AdjacencyPipe):
    """
    MTCoh pipe for spectral coherence estimation using
    multitaper methods

    Parameters
    ----------
        time_band: float
            The time half bandwidth resolution of the estimate [-NW, NW];
            such that resolution is 2*NW

        n_taper: int
            Number of Slepian sequences to use (Usually < 2*NW-1)

        cf: list
            Frequency range over which to compute coherence [-NW+C, C+NW]
    """

    def __init__(self, time_band, n_taper, cf):
        # Standard param checks
        check_type(time_band, float)
        check_type(n_taper, int)
        check_type(cf, list)
        if n_taper >= 2*time_band:
            raise Exception('Number of tapers must be less than 2*time_band')
        if not len(cf) == 2:
            raise Exception('Must give a frequency range in list of length 2')

        # Assign instance parameters
        self.time_band = time_band
        self.n_taper = n_taper
        self.cf = cf

    def _pipe_as_flow(self, signal_packet):
        # Get signal_packet details
        hkey = signal_packet.keys()[0]
        ax_0_ix = signal_packet[hkey]['meta']['ax_0']['index']
        ax_1_ix = signal_packet[hkey]['meta']['ax_1']['index']
        signal = signal_packet[hkey]['data']
        fs = np.int(np.mean(1./np.diff(ax_0_ix)))

        # Assume undirected connectivity
        triu_ix, triu_iy = np.triu_indices(len(ax_1_ix), k=1)

        # Initialize association matrix
        adj = np.zeros((len(ax_1_ix), len(ax_1_ix)))

        # Compute all coherences
        for n1, n2 in zip(triu_ix, triu_iy):
            out = mt_coherence(1.0/fs,
                               signal[:, n1],
                               signal[:, n2],
                               self.time_band,
                               self.n_taper,
                               int(len(ax_0_ix)/2.), 0.95,
                               iadapt=0,
                               cohe=True, freq=True)

            # Find closest frequency to the desired center frequency
            #cf_idx = np.argmin(np.abs(out['freq'] - self.cf))
            cf_idx = np.flatnonzero((out['freq'] >= self.cf[0]) &
                                    (out['freq'] <= self.cf[1]))

            # Store coherence in association matrix
            adj[n1, n2] = np.mean(out['cohe'][cf_idx])
        adj += adj.T

        new_packet = {}
        new_packet[hkey] = {
            'data': adj,
            'meta': {
                'ax_0': signal_packet[hkey]['meta']['ax_1'],
                'ax_1': signal_packet[hkey]['meta']['ax_1'],
                'time': {
                    'label': 'Time (sec)',
                    'index': np.float(ax_0_ix[-1])
                }
            }
        }

        return new_packet
