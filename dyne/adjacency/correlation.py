"""
Correlation pipes for quantifying signal similarity (i.e. connectivity)

Created by: Ankit Khambhati

Change Log
----------
2016/03/06 - Implemented XCorr and Corr pipes
"""

from __future__ import division
import numpy as np

from errors import check_type
from base import AdjacencyPipe


class XCorr(AdjacencyPipe):
    """
    XCorr pipe for cross-correlation association between signals

    This class implements an FFT-based cross-correlation.
    """

    def __init__(self):
        self = self

    def _pipe_as_flow(self, signal_packet):
        # Get signal_packet details
        hkey = signal_packet.keys()[0]
        ax_0_ix = signal_packet[hkey]['meta']['ax_0']['index']
        ax_1_ix = signal_packet[hkey]['meta']['ax_1']['index']
        signal = signal_packet[hkey]['data']

        # Assume undirected connectivity
        triu_ix, triu_iy = np.triu_indices(len(ax_1_ix), k=1)

        # Normalize the signal
        signal -= signal.mean(axis=0)
        signal /= signal.std(axis=0)

        # Initialize adjacency matrix
        adj = np.zeros((len(ax_1_ix), len(ax_1_ix)))

        #assoc = np.abs(np.corrcoef(signal.T))
        # Use FFT to compute cross-correlation
        signal_fft = np.fft.rfft(
            np.vstack((signal, np.zeros_like(signal))),
            axis=0)

        # Iterate over all edges
        for n1, n2 in zip(triu_ix, triu_iy):
            xc = 1 / len(ax_0_ix) * np.fft.irfft(
                signal_fft[:, n1] * np.conj(signal_fft[:, n2]))
            adj[n1, n2] = np.max(np.abs(xc))
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


class Corr(AdjacencyPipe):
    """
    Corr pipe for Pearson correlation association between signals

    This class implements a standard Pearson correlation measure.
    """

    def __init__(self):
        self = self

    def _pipe_as_flow(self, signal_packet):
        # Get signal_packet details
        hkey = signal_packet.keys()[0]
        signal = signal_packet[hkey]['data']

        # Apply Pearson correlation
        adj = np.abs(np.corrcoef(signal.T))

        new_packet = {}
        new_packet[hkey] = {
            'data': adj,
            'meta': {
                'ax_0': signal_packet[hkey]['meta']['ax_1'],
                'ax_1': signal_packet[hkey]['meta']['ax_1'],
                'time': {
                    'label': 'Time (sec)',
                    'index': np.float(
                        signal_packet[hkey]['meta']['ax_0']['index'][-1])
                }
            }
        }

        return new_packet
