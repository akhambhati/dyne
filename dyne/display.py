'''
Display messages to the console or screen

Created by: Ankit Khambhati

Change Log
----------
2016/02/22 - Consolidate display code from other local modules
'''

import sys


def my_display(txt):
    '''
    Print string to the screen

    Parameters
    ----------
        txt: str
            Text to be printed to the screen

        verbose: bool
            Option to robustly automate whether
            text gets printed to the screen
    '''

    sys.stdout.write(txt)
    sys.stdout.flush()
