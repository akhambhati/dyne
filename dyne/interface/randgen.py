"""
Test sequences for simulating
InterfacePipe functionality in DyNe
"""

import numpy as np

from display import my_display
from errors import check_type
from base import InterfacePipe

__all__ = ['MvarNormalNoise']

class MvarNormalNoise(InterfacePipe):
    """
    Generate noise signal from an underlying
    multivariate normal distribution with specific co-variance structure

    Parameters
    ----------
        n_node: int
            Number of nodes to simulate
        n_sample: int
            Number of samples to simulate
        win_width: int
            Number of samples in a window (signal_packet)
        win_shift: int
            Number of samples to shift the window (signal_packet)

    Yields
    ------
        signal_packet (see InterfacePipe documentation)
    """

    def __init__(self, n_node, n_sample, win_width, win_shift):
        # Standard param checks
        check_type(n_node, int)
        check_type(n_sample, int)
        check_type(win_width, int)
        check_type(win_shift, int)

        # Assign to instance
        self.n_node = n_node
        self.n_sample = n_sample
        self.win_width = win_width
        self.win_shift = win_shift

    def _pipe_as_source(self):
        rnd_norm_matr = np.random.randn(self.n_node, self.n_node)
        rnd_cov_matr = np.abs(np.triu(rnd_norm_matr) +
                              np.triu(rnd_norm_matr).T)
        signal = np.random.multivariate_normal(np.zeros(self.n_node),
                                               rnd_cov_matr,
                                               self.n_sample)

        start_ix = 0
        while True:
            clip = signal[start_ix:start_ix+self.win_width, :]

            # Properly format the yield
            signal_packet = {}
            signal_packet['data'] = clip
            signal_packet['meta'] = \
                    {'ax_0':
                     {'label': 'Samples',
                      'index': np.arange(start_ix, start_ix+self.win_width)},
                     'ax_1':
                     {'label': 'Nodes',
                      'index': np.array(map(str, xrange(self.n_node)))}}
            yield signal_packet

            # Advance the pointer
            start_ix += self.win_shift

            if start_ix+self.win_width > self.n_sample:
                my_display('\nEnd of signal.\n')
                break
