"""
Console pipes for printing payload
to screen

Created by: Ankit Khambhati

Change Log
----------
2016/03/06 - Implemented Console pipe
"""

import numpy as np

from display import my_display, pprint
from errors import check_type
from base import LoggerPipe


class Console(LoggerPipe):
    """
    Display pipeline payload to screen

    Parameters
    ----------
        description: str
            Briefly describe the contents to follow (1-5 words)
    """

    def __init__(self, description):
        # Standard param checks
        check_type(description, str)

        # Assign to instance
        self.description = description

    def _pipe_as_flow(self, signal_packet):
        pp = pprint.PrettyPrinter(indent=len(self.description))

        console_str = '\n{}: \n{}'.format(
            self.description,
            pp.pformat(signal_packet))
        my_display(console_str)
