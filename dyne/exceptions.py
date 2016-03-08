"""
Exceptions and custom warnings for Dyne
"""

class PipeTypeError(TypeError):
    """
    Exception class if pipe type is not of a registered type
    """

class PipeLinkError(AttributeError, TypeError):
    """
    Exception class if a pipe is unlinked or cannot be linked to another pipe
    """
