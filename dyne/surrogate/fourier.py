"""
Classes for Fourier-based phase randomization surrogate generation
"""
# Author: Ankit Khambhati
# License: BSD 3-Clause


from __future__ import division
from ...common.pipe import SurrogatePipe
import numpy as np
import scipy.fftpack


class PhaseShuffle(SurrogatePipe):
    """
    PhaseShuffle class for randomizing phase relationships in incoming
    multivariate signal

    This class performs a FFT, randomizes phase, and performs an inverse FFT.

    Parameters
    ----------
    n_total_surrogate: int
        Number of surrogate signals to generate
    """

    def __init__(self, pipe_name=None, n_total_surrogate=None,
                 cache=None):
        super(PhaseShuffle, self).__init__(pipe_name, cache)
        assert type(n_total_surrogate) == int
        self.n_total_surrogate = n_total_surrogate

    def _func_def(self, signal_packet):
        """
        Apply Fourier phase randomization

        Parameters
        ----------
        signal_packet: dict
            SEE SIGPROC

        Attributes
        ----------
        b_, a_: list(float), list(float)
            Cached numerator, denominator filter coefficients

        Returns
        -------
        signal_packet: dict
            SEE SIGPROC
        """
        win = signal_packet['win']

        n_sample_win = win.shape[0]
        n_node = win.shape[1]

        # Find the middle point of the samples
        if np.mod(n_sample_win, 2) == 0:
            n_half_win = n_sample_win / 2
        else:
            n_half_win = (n_sample_win - 1) / 2

        win_fft = scipy.fftpack.fft(win, 2*n_half_win, axis=0)
        mag_ = np.abs(win_fft)
        ang_ = np.angle(win_fft)

        # Loop through surrogates
        win_ifft = np.zeros((self.n_total_surrogate, n_sample_win, n_node))
        for ns in xrange(self.n_total_surrogate):
            # New phases
            rand_ang = ang_[:n_half_win-1, :] + \
                np.random.uniform(0, 2*np.pi, size=(n_half_win-1, n_node))

            # Reconstruct randomized phase array for iFFT
            ifft_ang = np.zeros((n_sample_win, n_node))
            ifft_ang[0, :] = 0
            ifft_ang[1:n_half_win, :] = rand_ang
            ifft_ang[n_half_win, :] = ang_[n_half_win, :]
            ifft_ang[n_half_win+1:, :] = -1*rand_ang[::-1, :]

            # Reconstruct original magnitude spectrum
            ifft = np.zeros((n_sample_win, n_node))
            ifft[:n_half_win, :] = mag_[:n_half_win, :]
            ifft[n_half_win:, :] = mag_[n_half_win+1:1:-1, :]

            # Complex fourier coefficients (phaser form)
            win_ifft[ns, :, :] = ifft * np.exp(1j*ifft_ang)

        # Perform filtering and dump into signal_packet
        signal_packet['surr_win'] = np.real(scipy.fftpack.ifft(
            win_ifft, n_sample_win, axis=1))

        return signal_packet


class SignalShuffle(SurrogatePipe):
    """
    SignalShuffle class for randomizing relationships in incoming
    multivariate signal

    This class permutes the signal time-series.

    Parameters
    ----------
    n_total_surrogate: int
        Number of surrogate signals to generate
    """

    def __init__(self, pipe_name=None, n_total_surrogate=None,
                 cache=None):
        super(SignalShuffle, self).__init__(pipe_name, cache)
        assert type(n_total_surrogate) == int
        self.n_total_surrogate = n_total_surrogate

    def _func_def(self, signal_packet):
        """
        Apply Permutation randomization

        Parameters
        ----------
        signal_packet: dict
            SEE SIGPROC

        Returns
        -------
        signal_packet: dict
            SEE SIGPROC
        """
        win = signal_packet['win']

        surr_win = np.zeros((self.n_total_surrogate,
                             win.shape[0],
                             win.shape[1]))

        for r in xrange(self.n_total_surrogate):
            surr_win[r, :, :] = np.array(map(np.random.permutation, win.T)).T

        # Dump into signal_packet
        signal_packet['surr_win'] = surr_win

        return signal_packet


class CircShuffle(SurrogatePipe):
    """
    CircShuffle class for randomizing phase relationships in incoming
    multivariate signal


    Parameters
    ----------
    n_total_surrogate: int
        Number of surrogate signals to generate
    """

    def __init__(self, pipe_name=None, n_total_surrogate=None,
                 cache=None):
        super(CircShuffle, self).__init__(pipe_name, cache)
        assert type(n_total_surrogate) == int
        self.n_total_surrogate = n_total_surrogate

    def _func_def(self, signal_packet):
        """
        Apply Circular shuffle

        Parameters
        ----------
        signal_packet: dict
            SEE SIGPROC

        Returns
        -------
        signal_packet: dict
            SEE SIGPROC
        """
        win = signal_packet['win']
        fs = signal_packet['fs']

        n_sample_win = win.shape[0]
        n_node = win.shape[1]

        surr_win = np.zeros((self.n_total_surrogate,
                             n_sample_win,
                             n_node))

        shift = np.random.randint(low=0, high=int(n_sample_win),
                                  size=(self.n_total_surrogate, n_node))
        for r in xrange(self.n_total_surrogate):
            for node in xrange(n_node):
                surr_win[r, :, node] = np.roll(win[:, node], shift[r, node])

        # Dump into signal_packet
        signal_packet['surr_win'] = surr_win

        return signal_packet
