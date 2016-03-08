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

from display import my_display, pprint
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

    Attributes
    ----------
    b_: array
        IIR filter numerator coefficients

    a_: array
        IIR filter denominator coefficients
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
