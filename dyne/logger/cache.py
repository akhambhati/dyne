"""
Cache pipes for saving pipeline payload to disk

Created by: Ankit Khambhati

Change Log
----------
2016/03/06 - Implemented SaveHDF pipe
"""

import numpy as np
import h5py

from display import my_display, pprint
from errors import check_type
from base import LoggerPipe


class SaveHDF(LoggerPipe):
    """
    Save pipeline payload as an HDF

    Parameters
    ----------
        cache_name: str
            Name of HDF cache file
    """

    def __init__(self, cache_name):

        # Standard param checks
        check_type(cache_name, str)

        # Assign to instance
        self.cache_name = cache_name
        self.new_hdf = True


    def _pipe_as_flow(self, signal_packet):

        if self.new_hdf:
            df = h5py.File('{}'.format(self.cache_name), 'w')
            self.new_hdf = False
        else:
            df = h5py.File('{}'.format(self.cache_name), 'a')

        def walk(dd, df):
            for key, value in dd.iteritems():
                if isinstance(value, dict):
                    try:
                        dset = df[key]
                    except:
                        dset = df.require_group(key)
                    walk(value, dset)
                else:
                    if (type(value) is np.float) or \
                       (type(value) is np.int):
                        try:
                            dset = df[key]
                        except KeyError:
                            dset = df.require_dataset(
                                key,
                                (0, 1),
                                type(value),
                                maxshape=(None, 1),
                                compression='lzf')
                        dset.resize(dset.shape[0]+1, axis=0)
                        dset[-1, 0] = value

                    if (type(value) is np.str):
                        try:
                            dset = df[key]
                        except KeyError:
                            dt = h5py.special_dtype(vlen=unicode)
                            dset = df.require_dataset(
                                key,
                                (0, 1),
                                dt,
                                maxshape=(None, 1),
                                compression='lzf')
                        dset.resize(dset.shape[0]+1, axis=0)
                        dset[-1, 0] = value

                    if type(value) is np.ndarray:
                        if type(value[0]) is np.string_:
                            dt = h5py.special_dtype(vlen=unicode)
                        else:
                            dt = np.float
                        try:
                            dset = df[key]
                        except KeyError:
                            dset = df.require_dataset(
                                key,
                                (0,)+value.shape,
                                dt,
                                maxshape=(None,)+value.shape,
                                compression='lzf')
                        dset.resize(dset.shape[0]+1, axis=0)
                        dset[-1, ...] = value
        walk(signal_packet, df)
        df.flush()
        df.close()
