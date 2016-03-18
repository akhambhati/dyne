"""
Offline pipes for streaming in cached data

Created by: Ankit Khambhati

Change Log
----------
2016/03/18 - Implemented CSVSignal pipe
2016/03/10 - Implemented MATSignal pipe
"""

import os.path
import numpy as np
import h5py

from ..errors import check_type, check_path, check_has_key
from ..base import InterfacePipe


class MATSignal(InterfacePipe):
    """
    MATSignal pipe for interfacing neural signals stored in formatted MAT files

    This class interfaces external multivariate signals in MATLAB (v7.3) to the
    dyne framework.

    Parameters
    ----------
        signal_path: str
            Path to MAT-file containing neural signals

        win_len: float
            Time length of windows

        win_disp: float
            Time displacement of consecutive windows

    """

    def __init__(self, signal_path, win_len, win_disp):
        # Standard param checks
        check_type(signal_path, str)
        check_type(win_len, float)
        check_type(win_disp, float)
        check_path(signal_path, exist=True)
        if win_disp > win_len:
            raise ValueError('win_len cannot be shorter than win_disp')

        # Assign to instance
        self.signal_path = signal_path
        self.win_len = win_len
        self.win_disp = win_disp

        # Open the MAT file and make sure all information is available
        self._cache_signal()

    def _cache_signal(self):
        # Open file using appropriate io utility
        try:
            df_signal = h5py.File(self.signal_path, 'r')
        except:
            raise IOError('Could not load %s with h5py' %
                          self.signal_path)

        # Check all components in place
        check_has_key(df_signal, 'evData')
        check_has_key(df_signal, 'Fs')
        check_has_key(df_signal, 'channels')

        # Ensure evData is properly formatted
        if not len(df_signal['evData'].shape) == 2:
            raise ValueError('evData should have 2 dimensions')
        try:
            iter(df_signal['evData'])
        except TypeError:
            raise TypeError('evData must be iterable')
        self.signal_ = df_signal['evData']
        self.n_node_ = df_signal['evData'].shape[1]
        self.n_sample_ = df_signal['evData'].shape[0]

        # Get sampling frequency
        if not np.dtype(df_signal['Fs']) == np.float64:
            raise TypeError('Fs should be a float64')
        self.sample_frequency_ = df_signal['Fs'][0, 0]

        # Get channel labels
        if not df_signal['channels'].shape[0] < \
           df_signal['channels'].shape[1]:
            raise ValueError('Channels improperly stored, try transposing')
        self.node_ = np.array(
            [u''.join(unichr(c) for c in df_signal[obj_ref])
             for obj_ref in df_signal['channels'][0, :]], dtype=np.str)

        # Check window size and displacement
        self.n_win_len = int(self.win_len * self.sample_frequency_)
        self.n_win_disp = int(self.win_disp * self.sample_frequency_)
        if self.n_win_len > self.n_sample_:
            raise ValueError('win_len cannot be longer than signal duration')
        self.n_wins = ((self.n_sample_ - self.n_win_len) /
                       self.n_win_disp) + 1

    def _pipe_as_source(self):
        for idx in range(0,
                         self.n_wins*self.n_win_disp,
                         self.n_win_disp):

            # window nan adjust
            win = self.signal_[idx:idx+self.n_win_len, :]
            nan_idx = np.nonzero(np.isnan(win))
            win[nan_idx[0], nan_idx[1]] = np.nanmean(win[:, nan_idx[1]], axis=0)

            # Format the signal_packet
            signal_packet = {}
            signal_packet['data'] = win
            signal_packet['meta'] = \
                {'ax_0':
                 {'label': 'Time (sec)',
                  'index': np.arange(idx, idx+self.n_win_len) /
                                self.sample_frequency_},
                 'ax_1':
                 {'label': 'Nodes',
                  'index': self.node_}
                }

            yield signal_packet


class CSVSignal(InterfacePipe):
    """
    CSVSignal pipe for interfacing neural signals stored in formatted CSV files

    This class interfaces external multivariate signals in CSV to the
    dyne framework.

    Parameters
    ----------
        signal_path: str
            Path to CSV-file containing neural signals

        sample_frequency_: float
            Sampling frequency of signals

        win_len: float
            Time length of windows

        win_disp: float
            Time displacement of consecutive windows
    """

    def __init__(self, signal_path, sample_frequency, win_len, win_disp):
        # Standard param checks
        check_type(signal_path, str)
        check_type(sample_frequency, float)
        check_type(win_len, float)
        check_type(win_disp, float)
        check_path(signal_path, exist=True)
        if win_disp > win_len:
            raise ValueError('win_len cannot be shorter than win_disp')

        # Assign to instance
        self.signal_path = signal_path
        self.sample_frequency = sample_frequency
        self.win_len = win_len
        self.win_disp = win_disp

        # Open the CSV file and make sure all information is available
        self._cache_signal()

    def _cache_signal(self):
        # Open file using appropriate io utility
        try:
            evData = np.array(np.genfromtxt(self.signal_path, delimiter=','),
                              dtype=np.float)
        except:
            raise IOError('Could not load %s with numpy.genfromtxt' %
                          self.signal_path)

        # Ensure evData is properly formatted
        if not len(evData.shape) == 2:
            raise ValueError('evData should have 2 dimensions')
        try:
            iter(evData)
        except TypeError:
            raise TypeError('evData must be iterable')
        self.signal_ = evData
        self.n_node_ = evData.shape[1]
        self.n_sample_ = evData.shape[0]

        # Get channel labels
        self.node_ = np.arange(self.n_node_) + 1

        # Check window size and displacement
        self.n_win_len = int(self.win_len * self.sample_frequency)
        self.n_win_disp = int(self.win_disp * self.sample_frequency)
        if self.n_win_len > self.n_sample_:
            raise ValueError('win_len cannot be longer than signal duration')
        self.n_wins = ((self.n_sample_ - self.n_win_len) /
                       self.n_win_disp) + 1

    def _pipe_as_source(self):
        for idx in range(0,
                         self.n_wins*self.n_win_disp,
                         self.n_win_disp):

            # window nan adjust
            win = self.signal_[idx:idx+self.n_win_len, :]
            nan_idx = np.nonzero(np.isnan(win))
            win[nan_idx[0], nan_idx[1]] = np.nanmean(win[:, nan_idx[1]], axis=0)

            # Format the signal_packet
            signal_packet = {}
            signal_packet['data'] = win
            signal_packet['meta'] = \
                {'ax_0':
                 {'label': 'Time (sec)',
                  'index': np.arange(idx, idx+self.n_win_len) /
                                self.sample_frequency},
                 'ax_1':
                 {'label': 'Nodes',
                  'index': self.node_}
                }

            yield signal_packet

