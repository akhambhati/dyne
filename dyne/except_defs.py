"""
Define exceptions and warnings for DyNe

Created by: Ankit Khambhati

Change Log
----------
2016/03/10 - Generated __all__ definition
"""

__all__ = ['PipeTypeError',
           'PipeLinkError']


class PipeTypeError(TypeError):
    """
    Exception class if pipe type is not of a registered type
    """

class PipeLinkError(AttributeError, TypeError):
    """
    Exception class if a pipe is unlinked or cannot be linked to another pipe
    """
