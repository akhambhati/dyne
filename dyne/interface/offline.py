"""
SOURCE: Offline signal sources class for interfacing sensor data with dyne
"""

import os.path
import numpy as np
import h5py

from ..base import InterfacePipe


class MATSignal(InterfacePipe):
    """
    MATSignal class for interfacing MATLAB multivariate signals in dyne

    This class interfaces external multivariate signals in MATLAB (v7.3) to the
    dyne framework.

    Parameters
    ----------
    signal_path: str
        Path or location to multivariate signal data

    info_path: str
        Path or location to global multivariate signal information

    win_len: float
        Time length of windows

    win_disp: float
        Time displacement of consecutive windows

    Attributes
    ----------
    n_node_: int
        Number of network nodes corresponding to number of sensors in signal

    n_sample_: int
        Number of samples in signal

    sample_frequency_: float
        Number of samples per second of signal

    signal_: ndarray, shape = [n_samples, n_nodes]
        Loaded multivariate signal

    nodes_: list(unicode)
        List of node names corresponding to sensor/channel names in signal

    n_sample_win: int
        Sample length of windows

    n_sample_disp: int
        Sample displacement of consecutive windows

    n_wins: int
        Total number of windows in the signal
    """

    def __init__(self, pipe_name=None, signal_path=None, info_path=None,
                 win_len=None, win_disp=None, cache=None):
        # Set input parameters
        super(MATSignal, self).__init__(pipe_name, cache)
        if not os.path.exists(signal_path):
            raise IOError('**ERROR** %s does not exist' % (signal_path))
        self.signal_path = signal_path

        self.info_path = info_path

        if win_disp > win_len:
            raise Exception('**ERROR** win_len (%0.2f) cannot be smaller than \
                            win_disp (%0.2f)' % (win_len, win_disp))
        self.win_len = win_len
        self.win_disp = win_disp

        # Initialize attributes
        self.n_node_ = None
        self.n_sample_ = None
        self.sample_frequency_ = None
        self.signal_ = None
        self.node_ = None

        self._cache_signal()

    def _cache_signal(self):
        # Open file using appropriate io utility
        try:
            df_signal = h5py.File(self.signal_path, 'r')
        except:
            raise Exception('**ERROR** Could not load %s' % (self.signal_type))
        try:
            df_info = h5py.File(self.info_path, 'r')
        except:
            pass

        # Get multivariate signal
        try:
            df_signal['evData']
            self.signal_ = df_signal['evData']
            assert len(df_signal['evData'].shape) == 2
            assert df_signal['evData'].shape[0] > df_signal['evData'].shape[1]
            self.n_node_ = df_signal['evData'].shape[1]
            self.n_sample_ = df_signal['evData'].shape[0]
        except KeyError:
            raise Exception('**ERROR** Signal does not contain evData')
        except AssertionError:
            raise Exception('**ERROR** Signal evData incorrectly formatted')
        try:
            iter(self.signal_)
        except TypeError:
            raise Exception("**ERROR** Signal type must be iterable")

        # Get sampling frequency
        try:
            df_signal['Fs']
            assert len(df_signal['Fs'].shape) == 2
            assert np.dtype(df_signal['Fs']) == np.float64
            self.sample_frequency_ = np.round(df_signal['Fs'][0, 0])
        except KeyError:
            raise Exception('**ERROR** Signal does not contain Fs')
        except AssertionError:
            raise Exception('**ERROR** Signal Fs is incorrectly formatted')

        # Get node labels from channel/sensor signal
        try:
            df_signal['channels']
            assert(df_signal['channels'].shape[0] <
                   df_signal['channels'].shape[1])
            self.node_ = [u''.join(unichr(c) for c in df_signal[obj_ref])
                          for obj_ref in df_signal['channels'][0, :]]
        except KeyError:
            self.node_ = [['CH%s' % idx] for idx in xrange(self.n_node_)]
            print(self.node_)
        except AssertionError:
            raise Exception(
                '**ERROR** Signal channels is incorrectly formatted')

        # Get bad node labels from channel/sensor signal
        try:
            df_info['excludeChannels']
            if len(df_info['excludeChannels'].shape) > 1:
                assert(df_info['excludeChannels'].shape[0] <
                       df_info['excludeChannels'].shape[1])
                bad_node_ = [u''.join(unichr(c) for c in df_info[obj_ref])
                             for obj_ref in df_info['excludeChannels'][0, :]]
                # Remove Bad Nodes
                node_idx = range(self.n_node_)
                for bnode in bad_node_:
                    try:
                        node_idx.pop(self.node_.index(bnode))
                        self.node_.remove(bnode)
                    except:
                        pass
                self.n_node_ = len(self.node_)
                self.signal_ = self.signal_[:, node_idx]
        except (NameError, KeyError, AssertionError):
            pass

        # Check window size and displacement
        self.n_sample_win = int(self.win_len * self.sample_frequency_)
        self.n_sample_disp = int(self.win_disp * self.sample_frequency_)

        if self.n_sample_win > self.n_sample_:
            raise Exception('**ERROR win_len (%0.2f) cannot be larger than \
                            signal duration (%0.2f)' %
                            (self.win_len,
                             self.n_samples / self.sample_frequency_))
        self.n_wins = ((self.n_sample_ - self.n_sample_win)
                       / self.n_sample_disp) + 1

    def _func_def(self):
        for idx in range(0,
                         self.n_wins*self.n_sample_disp,
                         self.n_sample_disp):

            # window nan adjust
            win = self.signal_[idx:idx+self.n_sample_win, :]
            nan_idx = np.nonzero(np.isnan(win))
            win[nan_idx[0], nan_idx[1]] = np.nanmean(win[:, nan_idx[1]], axis=0)

            signal_packet = {'idx': idx+self.n_sample_win,
                             'fs': self.sample_frequency_,
                             'win': win}
            yield signal_packet
