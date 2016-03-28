"""
Classes for defining the generic pipes for the pipeline

Created by: Ankit Khambhati

Change Log
----------
2016/03/28 - Added GlobalTopoPipe pipe types
2016/03/10 - Added NodeTopoPipe and EdgeTopoPipe pipe types
2016/03/08 - Added AdjacencyPipe pipe type
2016/03/08 - Added LoggerPipe, InterfacePipe, PreprocPipe pipe types
2016/03/02 - Established the BasePipe
"""

import numpy as np

import json
import hashlib
from functools import wraps
import inspect
import copy

import display
import except_defs as exceptions
import errors


def coroutine(func):
    """Advance a coroutine to first yield point"""

    @wraps(func)
    def start(*args, **kwargs):
        cr = func(*args, **kwargs)
        next(cr)
        return cr
    return start


class BasePipe(object):
    """
    Base class for all pipes in DyNe

    Rules
    -----
        1. All derived pipe classes must explicitly define all pipe parameters
           as variable names in __init__
        2. No *args or **kwargs may be used
    """

    @classmethod
    def _get_param_var(cls):
        """Return parameters specified by __init__ method of the class"""

        all_param_var = inspect.getargspec(cls.__init__)[0]
        return [param for param in all_param_var if not param == 'self']

    def to_JSON(self):
        """Return JSON dictionary of pipe parameters"""

        param_dict = {}
        for param in self._get_param_var():
            param_dict[param] = self.__dict__[param]
        return json.dumps(param_dict, sort_keys=True)

    def to_hash(self):
        """Return hashtag identifier of pipe parameters"""
        fmt_str = '{}.{}: {}'.format(
            self.__module__,
            self.__class__.__name__,
            self.to_JSON())
        return hashlib.sha224(fmt_str).hexdigest()

    def _tag_signal_packet(self, signal_packet):
        """Tag the signal packet before sending it downstream"""
        new_packet = {}
        new_packet[self.to_hash()] = signal_packet

        return new_packet

    def _retag_signal_packet(self, signal_packet):
        """Replace the tag on signal packet before sending it downstream"""
        new_packet = {}
        new_packet[self.to_hash()] = \
            signal_packet[signal_packet.keys()[0]]

        return new_packet

    @classmethod
    def get_valid_link(self):
        """Return list of pipe types the current pipe type can link to"""
        raise NotImplementedError('Must return list of pipe types' +
                                  ' the current pipe type can connect to')

    def link(self, downstream_pipe_list):
        """
        Link the current pipe to a downstream pipe
        """
        # Standard param check
        errors.check_type(downstream_pipe_list, list)

        self.downstream_pipe_flow = []
        for downstream_pipe in downstream_pipe_list:
            if downstream_pipe:
                check_pipe_type = [isinstance(downstream_pipe, val_pipe_type)
                                   for val_pipe_type in self.get_valid_link()]
                if not any(check_pipe_type):
                    raise exceptions.PipeLinkError(
                        '%r must be one of the following pipe types: %r' %
                        (downstream_pipe, self.get_valid_link()))
                self.downstream_pipe_flow.append(
                    downstream_pipe.apply_pipe_as_flow())

    def apply_pipe_as_source(self):
        """Run the pipe as a source"""
        try:
            gen = self._pipe_as_source()
        except AttributeError:
            raise NotImplementedError(
                '%r does not have _pipe_as_source implemented' %
                self.__class__.__name__)

        while True:
            try:
                signal_packet = gen.next()
            except StopIteration:
                display.my_display('\nClosing Source Pipe: %r' %
                                   self.__class__.__name__)
                break

            signal_packet = self._tag_signal_packet(signal_packet)
            self._verify_signal_packet(signal_packet)

            try:
                self.downstream_pipe_flow
            except AttributeError:
                raise exceptions.PipeLinkError(
                    '%r must link to downstream pipe using link() method' %
                    self.__class__.__name__)

            for downstream_pipe in self.downstream_pipe_flow:
                downstream_pipe.send(signal_packet)

        gen.close()
        for downstream_pipe in self.downstream_pipe_flow:
            downstream_pipe.close()
        display.my_display('\n')

    @coroutine
    def apply_pipe_as_flow(self):
        """Run the pipe as a flow"""
        try:
            self._pipe_as_flow
        except AttributeError:
            raise NotImplementedError(
                '%r does not have _pipe_as_flow implemented' %
                self.__class__.__name__)

        while True:
            try:
                signal_packet = (yield)
            except GeneratorExit:
                display.my_display('\nClosing Flow Pipe: %r' %
                                   self.__class__.__name__)
                break

            signal_packet = self._pipe_as_flow(copy.deepcopy(signal_packet))
            if signal_packet:
                signal_packet = self._retag_signal_packet(signal_packet)
                self._verify_signal_packet(signal_packet)

            try:
                self.downstream_pipe_flow
            except AttributeError:
                raise exceptions.PipeLinkError(
                    '%r must link to downstream pipe using link() method' %
                    self.__class__.__name__)

            for downstream_pipe in self.downstream_pipe_flow:
                downstream_pipe.send(signal_packet)

        for downstream_pipe in self.downstream_pipe_flow:
            downstream_pipe.close()

    def _verify_signal_packet(self, signal_packet):
        """Signal packet must follow pipe type organization"""
        raise NotImplementedError(
            '%r does not have _verify_signal_packet implemented' %
            self.__class__.__name__)


class LoggerPipe(BasePipe):
    """
    Pipe Type: LoggerPipe

    Consumes
    --------
        signal_packet: dict
            1) hashkey: dict
                A) Arbitrary organization

    Linkable pipe types:
        None
    """

    def _verify_signal_packet(self, signal_packet):
        """Ensure signal packet is organized properly"""
        errors.check_type(signal_packet, dict)
        if len(signal_packet.keys()) > 1:
            raise ValueError('signal_packet base-level should contain only' +
                             ' the pipe hash identifier as key')

    def get_valid_pipe(self):
        return []


class InterfacePipe(BasePipe):
    """
    Pipe Type: InterfacePipe

    Accepts
    -------
        [Only acts as SOURCE]

    Yields
    ------
        signal_packet: dict
            1) hashkey: dict
                A) data: numpy.ndarray, shape: [n_sample x n_node]
                    Windowed signal
                B) meta: dict
                    i. ax_0: dict
                        a. label: str
                            Describes unit of measurement for n_sample
                        b. index: numpy.ndarray
                            Time stamp for each sample
                    ii. ax_1: dict
                        a. label: str
                            Describes what n_node represents
                        b. index: numpy.ndarray
                            String label for each node

    Linkable pipe types:
        None
        LoggerPipe
        PreprocPipe
        AdjacencyPipe
    """

    def _verify_signal_packet(self, signal_packet):
        """Ensure signal packet is organized properly"""
        errors.check_type(signal_packet, dict)
        if len(signal_packet.keys()) > 1:
            raise ValueError('signal_packet base-level should contain only' +
                             ' the pipe hash identifier as key')
        hkey = signal_packet.keys()[0]

        errors.check_has_key(signal_packet[hkey], 'data')
        errors.check_has_key(signal_packet[hkey], 'meta')
        errors.check_has_key(signal_packet[hkey]['meta'], 'ax_0')
        errors.check_has_key(signal_packet[hkey]['meta'], 'ax_1')
        errors.check_has_key(signal_packet[hkey]['meta']['ax_0'], 'label')
        errors.check_has_key(signal_packet[hkey]['meta']['ax_0'], 'index')
        errors.check_has_key(signal_packet[hkey]['meta']['ax_1'], 'label')
        errors.check_has_key(signal_packet[hkey]['meta']['ax_1'], 'index')

        errors.check_type(signal_packet[hkey]['data'], np.ndarray)
        errors.check_type(signal_packet[hkey]['meta']['ax_0']['label'], str)
        errors.check_type(signal_packet[hkey]['meta']['ax_0']['index'],
                          np.ndarray)
        errors.check_type(signal_packet[hkey]['meta']['ax_1']['label'], str)
        errors.check_type(signal_packet[hkey]['meta']['ax_1']['index'],
                          np.ndarray)

    def get_valid_link(self):
        return [LoggerPipe,
                PreprocPipe,
                AdjacencyPipe]


class PreprocPipe(BasePipe):
    """
    Pipe Type: PreprocPipe

    Accepts
    -------
         signal_packet: dict
            1) hashkey: dict
                A) data: numpy.ndarray, shape: [n_sample x n_node]
                    Windowed signal
                B) meta: dict
                    i. ax_0: dict
                        a. label: str
                            Describes unit of measurement for n_sample
                        b. index: numpy.ndarray
                            Time stamp for each sample
                    ii. ax_1: dict
                        a. label: str
                            Describes what n_node represents
                        b. index: numpy.ndarray
                            String label for each node

    Yields
    ------
        signal_packet: dict
            1) hashkey: dict
                A) data: numpy.ndarray, shape: [n_sample x n_node]
                    Windowed signal
                B) meta: dict
                    i. ax_0: dict
                        a. label: str
                            Describes unit of measurement for n_sample
                        b. index: numpy.ndarray
                            Time stamp for each sample
                    ii. ax_1: dict
                        a. label: str
                            Describes what n_node represents
                        b. index: numpy.ndarray
                            String label for each node

    Linkable pipe types:
        None
        LoggerPipe
        PreprocPipe
        AdjacencyPipe
    """

    def _verify_signal_packet(self, signal_packet):
        """Ensure signal packet is organized properly"""
        errors.check_type(signal_packet, dict)
        if len(signal_packet.keys()) > 1:
            raise ValueError('signal_packet base-level should contain only' +
                             ' the pipe hash identifier as key')
        hkey = signal_packet.keys()[0]

        errors.check_has_key(signal_packet[hkey], 'data')
        errors.check_has_key(signal_packet[hkey], 'meta')
        errors.check_has_key(signal_packet[hkey]['meta'], 'ax_0')
        errors.check_has_key(signal_packet[hkey]['meta'], 'ax_1')
        errors.check_has_key(signal_packet[hkey]['meta']['ax_0'], 'label')
        errors.check_has_key(signal_packet[hkey]['meta']['ax_0'], 'index')
        errors.check_has_key(signal_packet[hkey]['meta']['ax_1'], 'label')
        errors.check_has_key(signal_packet[hkey]['meta']['ax_1'], 'index')

        errors.check_type(signal_packet[hkey]['data'], np.ndarray)
        errors.check_type(signal_packet[hkey]['meta']['ax_0']['label'], str)
        errors.check_type(signal_packet[hkey]['meta']['ax_0']['index'],
                          np.ndarray)
        errors.check_type(signal_packet[hkey]['meta']['ax_1']['label'], str)
        errors.check_type(signal_packet[hkey]['meta']['ax_1']['index'],
                          np.ndarray)

    def get_valid_link(self):
        return [LoggerPipe,
                PreprocPipe,
                AdjacencyPipe]


class AdjacencyPipe(BasePipe):
    """
    Pipe Type: AdjacencyPipe

    Accepts
    -------
         signal_packet: dict
            1) hashkey: dict
                A) data: numpy.ndarray, shape: [n_sample x n_node]
                    Windowed signal
                B) meta: dict
                    i. ax_0: dict
                        a. label: str
                            Describes unit of measurement for n_sample
                        b. index: numpy.ndarray
                            Time stamp for each sample
                    ii. ax_1: dict
                        a. label: str
                            Describes what n_node represents
                        b. index: numpy.ndarray
                            String label for each node

    Yields
    ------
        signal_packet: dict
            1) hashkey: dict
                A) data: numpy.ndarray, shape: [n_node x n_node]
                    Connectivity between nodes
                B) meta: dict
                    i. ax_0: dict
                        a. label: str
                            Describes what n_node represents
                        b. index: numpy.ndarray
                            String label for each node
                    ii. ax_1: dict
                        a. label: str
                            Describes what n_node represents
                        b. index: numpy.ndarray
                            String label for each node
                    iii. time: dict
                        a. label: str
                            Describes the unit of measurement
                        b. index: float
                            Timestamp represented by this packet

    Linkable pipe types:
        None
        LoggerPipe
    """

    def _verify_signal_packet(self, signal_packet):
        """Ensure signal packet is organized properly"""
        errors.check_type(signal_packet, dict)
        if len(signal_packet.keys()) > 1:
            raise ValueError('signal_packet base-level should contain only' +
                             ' the pipe hash identifier as key')
        hkey = signal_packet.keys()[0]

        errors.check_has_key(signal_packet[hkey], 'data')
        errors.check_has_key(signal_packet[hkey], 'meta')
        errors.check_has_key(signal_packet[hkey]['meta'], 'ax_0')
        errors.check_has_key(signal_packet[hkey]['meta'], 'ax_1')
        errors.check_has_key(signal_packet[hkey]['meta']['ax_0'], 'label')
        errors.check_has_key(signal_packet[hkey]['meta']['ax_0'], 'index')
        errors.check_has_key(signal_packet[hkey]['meta']['ax_1'], 'label')
        errors.check_has_key(signal_packet[hkey]['meta']['ax_1'], 'index')
        errors.check_has_key(signal_packet[hkey]['meta']['time'], 'label')
        errors.check_has_key(signal_packet[hkey]['meta']['time'], 'index')

        errors.check_type(signal_packet[hkey]['data'], np.ndarray)
        errors.check_type(signal_packet[hkey]['meta']['ax_0']['label'], str)
        errors.check_type(signal_packet[hkey]['meta']['ax_0']['index'],
                          np.ndarray)
        errors.check_type(signal_packet[hkey]['meta']['ax_1']['label'], str)
        errors.check_type(signal_packet[hkey]['meta']['ax_1']['index'],
                          np.ndarray)
        errors.check_type(signal_packet[hkey]['meta']['time']['label'], str)
        errors.check_type(signal_packet[hkey]['meta']['time']['index'], float)

    def get_valid_link(self):
        return [GlobalTopoPipe,
                NodeTopoPipe,
                EdgeTopoPipe,
                LoggerPipe]


class GlobalTopoPipe(BasePipe):
    """
    Pipe Type: GlobalTopoPipe

    Accepts
    -------
         signal_packet: dict
            1) hashkey: dict
                A) data: numpy.ndarray, shape: [n_node x n_node]
                    Connectivity between nodes
                B) meta: dict
                    i. ax_0: dict
                        a. label: str
                            Describes what n_node represents
                        b. index: numpy.ndarray
                            String label for each node
                    ii. ax_1: dict
                        a. label: str
                            Describes what n_node represents
                        b. index: numpy.ndarray
                            String label for each node
                    iii. time: dict
                        a. label: str
                            Describes the unit of measurement
                        b. index: float
                            Timestamp represented by this packet

    Yields
    ------
        signal_packet: dict
            1) hashkey: dict
                A) data: numpy.ndarray, shape: [1 x 1]
                    Global topology measurement
                B) meta: dict
                    i. time: dict
                        a. label: str
                            Describes the unit of measurement
                        b. index: float
                            Timestamp represented by this packet

    Linkable pipe types:
        None
        LoggerPipe
    """

    def _verify_signal_packet(self, signal_packet):
        """Ensure signal packet is organized properly"""
        errors.check_type(signal_packet, dict)
        if len(signal_packet.keys()) > 1:
            raise ValueError('signal_packet base-level should contain only' +
                             ' the pipe hash identifier as key')
        hkey = signal_packet.keys()[0]

        errors.check_has_key(signal_packet[hkey], 'data')
        errors.check_has_key(signal_packet[hkey], 'meta')
        errors.check_has_key(signal_packet[hkey]['meta']['time'], 'label')
        errors.check_has_key(signal_packet[hkey]['meta']['time'], 'index')

        errors.check_type(signal_packet[hkey]['data'], np.ndarray)
        errors.check_type(signal_packet[hkey]['meta']['time']['label'], str)
        errors.check_type(signal_packet[hkey]['meta']['time']['index'], float)

    def get_valid_link(self):
        return [LoggerPipe]


class NodeTopoPipe(BasePipe):
    """
    Pipe Type: NodeTopoPipe

    Accepts
    -------
         signal_packet: dict
            1) hashkey: dict
                A) data: numpy.ndarray, shape: [n_node x n_node]
                    Connectivity between nodes
                B) meta: dict
                    i. ax_0: dict
                        a. label: str
                            Describes what n_node represents
                        b. index: numpy.ndarray
                            String label for each node
                    ii. ax_1: dict
                        a. label: str
                            Describes what n_node represents
                        b. index: numpy.ndarray
                            String label for each node
                    iii. time: dict
                        a. label: str
                            Describes the unit of measurement
                        b. index: float
                            Timestamp represented by this packet

    Yields
    ------
        signal_packet: dict
            1) hashkey: dict
                A) data: numpy.ndarray, shape: [n_node x 1]
                    Node-based topology measurement
                B) meta: dict
                    i. ax_0: dict
                        a. label: str
                            Describes what n_node represents
                        b. index: numpy.ndarray
                            String label for each node
                    iii. time: dict
                        a. label: str
                            Describes the unit of measurement
                        b. index: float
                            Timestamp represented by this packet

    Linkable pipe types:
        None
        LoggerPipe
    """

    def _verify_signal_packet(self, signal_packet):
        """Ensure signal packet is organized properly"""
        errors.check_type(signal_packet, dict)
        if len(signal_packet.keys()) > 1:
            raise ValueError('signal_packet base-level should contain only' +
                             ' the pipe hash identifier as key')
        hkey = signal_packet.keys()[0]

        errors.check_has_key(signal_packet[hkey], 'data')
        errors.check_has_key(signal_packet[hkey], 'meta')
        errors.check_has_key(signal_packet[hkey]['meta'], 'ax_0')
        errors.check_has_key(signal_packet[hkey]['meta']['ax_0'], 'label')
        errors.check_has_key(signal_packet[hkey]['meta']['ax_0'], 'index')
        errors.check_has_key(signal_packet[hkey]['meta']['time'], 'label')
        errors.check_has_key(signal_packet[hkey]['meta']['time'], 'index')

        errors.check_type(signal_packet[hkey]['data'], np.ndarray)
        errors.check_type(signal_packet[hkey]['meta']['ax_0']['label'], str)
        errors.check_type(signal_packet[hkey]['meta']['ax_0']['index'],
                          np.ndarray)
        errors.check_type(signal_packet[hkey]['meta']['time']['label'], str)
        errors.check_type(signal_packet[hkey]['meta']['time']['index'], float)

    def get_valid_link(self):
        return [LoggerPipe]


class EdgeTopoPipe(BasePipe):
    """
    Pipe Type: EdgeTopoPipe

    Accepts
    -------
         signal_packet: dict
            1) hashkey: dict
                A) data: numpy.ndarray, shape: [n_node x n_node]
                    Connectivity between nodes
                B) meta: dict
                    i. ax_0: dict
                        a. label: str
                            Describes what n_node represents
                        b. index: numpy.ndarray
                            String label for each node
                    ii. ax_1: dict
                        a. label: str
                            Describes what n_node represents
                        b. index: numpy.ndarray
                            String label for each node
                    iii. time: dict
                        a. label: str
                            Describes the unit of measurement
                        b. index: float
                            Timestamp represented by this packet

    Yields
    ------
        signal_packet: dict
            1) hashkey: dict
                A) data: numpy.ndarray, shape: [n_node x n_node]
                    Connectivity between nodes
                B) meta: dict
                    i. ax_0: dict
                        a. label: str
                            Describes what n_node represents
                        b. index: numpy.ndarray
                            String label for each node
                    ii. ax_1: dict
                        a. label: str
                            Describes what n_node represents
                        b. index: numpy.ndarray
                            String label for each node
                    iii. time: dict
                        a. label: str
                            Describes the unit of measurement
                        b. index: float
                            Timestamp represented by this packet

    Linkable pipe types:
        None
        LoggerPipe
    """

    def _verify_signal_packet(self, signal_packet):
        """Ensure signal packet is organized properly"""
        errors.check_type(signal_packet, dict)
        if len(signal_packet.keys()) > 1:
            raise ValueError('signal_packet base-level should contain only' +
                             ' the pipe hash identifier as key')
        hkey = signal_packet.keys()[0]

        errors.check_has_key(signal_packet[hkey], 'data')
        errors.check_has_key(signal_packet[hkey], 'meta')
        errors.check_has_key(signal_packet[hkey]['meta'], 'ax_0')
        errors.check_has_key(signal_packet[hkey]['meta']['ax_0'], 'label')
        errors.check_has_key(signal_packet[hkey]['meta']['ax_0'], 'index')
        errors.check_has_key(signal_packet[hkey]['meta']['ax_1'], 'label')
        errors.check_has_key(signal_packet[hkey]['meta']['ax_1'], 'index')
        errors.check_has_key(signal_packet[hkey]['meta']['time'], 'label')
        errors.check_has_key(signal_packet[hkey]['meta']['time'], 'index')

        errors.check_type(signal_packet[hkey]['data'], np.ndarray)
        errors.check_type(signal_packet[hkey]['meta']['ax_0']['label'], str)
        errors.check_type(signal_packet[hkey]['meta']['ax_0']['index'],
                          np.ndarray)
        errors.check_type(signal_packet[hkey]['meta']['ax_1']['label'], str)
        errors.check_type(signal_packet[hkey]['meta']['ax_1']['index'],
                          np.ndarray)
        errors.check_type(signal_packet[hkey]['meta']['time']['label'], str)
        errors.check_type(signal_packet[hkey]['meta']['time']['index'], float)

    def get_valid_link(self):
        return [LoggerPipe]
