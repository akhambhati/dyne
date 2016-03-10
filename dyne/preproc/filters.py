"""
Filter pipes for conditioning pipeline payload

Created by: Ankit Khambhati

Change Log
----------
2016/03/06 - Implemented EllipticFilter, CommonAvgRef, Prewhiten pipes
"""

from __future__ import division
import numpy as np
import scipy.signal as spsig

from errors import check_type
from base import PreprocPipe


class EllipticFilter(PreprocPipe):
    """
    EllipticFilter pipe for bandpass, lowpass, highpass filtering

    This class implements zero-phase filtering to pre-process and analyze
    frequency-dependent network structure. Implements Elliptic IIR filter.

    Parameters
    ----------
        Wp: tuple, shape: (1,) or (1,1)
            Pass band cutoff frequency (Hz)

        Ws tuple, shape: (1,) or (1,1)
            Stop band cutoff frequency (Hz)

        Rp: float
            Pass band maximum loss (dB)

        As: float
            Stop band minimum attenuation (dB)
    """

    def __init__(self, Wp, Ws, Rp, As):
        # Standard param checks
        check_type(Wp, list)
        check_type(Ws, list)
        check_type(Rp, float)
        check_type(As, float)
        if not len(Wp) == len(Ws):
            raise Exception('Frequency criteria mismatch for Wp \
                            and Ws')
        if not ((len(Wp) == 1) or (len(Wp) == 2)):
            raise Exception('Must only be 1 or 2 frequency cutoffs \
                            in Wp and Ws')

        # Assign to instance
        self.Wp = Wp
        self.Ws = Ws
        self.Rp = Rp
        self.As = As

    def _pipe_as_flow(self, signal_packet):
        # Get signal_packet details
        hkey = signal_packet.keys()[0]
        ax_0_ix = signal_packet[hkey]['meta']['ax_0']['index']
        fs = np.int(np.mean(1./np.diff(ax_0_ix)))

        # Compute filter coefficients
        nyq = fs / 2.0
        wp_nyq = map(lambda f: f/nyq, self.Wp)
        ws_nyq = map(lambda f: f/nyq, self.Ws)
        coef_b, coef_a = spsig.iirdesign(wp=wp_nyq,
                                         ws=ws_nyq,
                                         gpass=self.Rp,
                                         gstop=self.As,
                                         analog=0, ftype='ellip',
                                         output='ba')

        # Perform filtering and dump into signal_packet
        signal_packet[hkey]['data'] = spsig.filtfilt(
            coef_b, coef_a, signal_packet[hkey]['data'], axis=0)

        return signal_packet


class CommonAvgRef(PreprocPipe):
    """
    CommonAvgRef pipe for removing the common-average from the signal

    """

    def __init__(self):
        self = self

    def _pipe_as_flow(self, signal_packet):
        # Get signal_packet details
        hkey = signal_packet.keys()[0]

        # Compute common average reference
        data = signal_packet[hkey]['data']
        data = (data.T - data.mean(axis=1)).T

        # Dump into signal_packet
        signal_packet[hkey]['data'] = data

        return signal_packet


class PreWhiten(PreprocPipe):
    """
    PreWhiten pipe for removing autocorrelative structure from each signal

    Implements an AR(1) filter and passes forth the residuals
    """

    def __init__(self):
        self = self

    def _pipe_as_flow(self, signal_packet):
        # Get signal_packet details
        hkey = signal_packet.keys()[0]
        ax_0_ix = signal_packet[hkey]['meta']['ax_0']['index']
        data = signal_packet[hkey]['data']

        win_white = np.zeros((data.shape[0]-1,
                              data.shape[1]))
        for i in xrange(data.shape[1]):
            win_x = np.vstack((data[:-1, i],
                               np.ones(data.shape[0]-1)))
            w = np.linalg.lstsq(win_x.T, data[1:, i])[0]
            win_white[:, i] = data[1:, i] - (data[:-1, i]*w[0] + w[1])

        ax_0_ix = ax_0_ix[1:]

        # Dump into signal_packet
        signal_packet[hkey]['data'] = win_white
        signal_packet[hkey]['meta']['ax_0']['index'] = ax_0_ix

        return signal_packet
