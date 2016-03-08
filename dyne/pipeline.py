"""
Classes for defining the generic pipes for the pipeline

Created by: Ankit Khambhati

Change Log
----------
2016/03/02 - Established the BasePipe
"""
# Author: Ankit Khambhati
# License: BSD 3-Clause

import json
import h5py
import importlib

import display
import errors

def _decode_list(data):
    rv = []
    for item in data:
        if isinstance(item, unicode):
            item = item.encode('utf-8')
        elif isinstance(item, list):
            item = _decode_list(item)
        elif isinstance(item, dict):
            item = _decode_dict(item)
        rv.append(item)
    return rv


def _decode_dict(data):
    rv = {}
    for key, value in data.iteritems():
        if isinstance(key, unicode):
            key = key.encode('utf-8')
        if isinstance(value, unicode):
            value = value.encode('utf-8')
        elif isinstance(value, list):
            value = _decode_list(value)
        elif isinstance(value, dict):
            value = _decode_dict(value)
        rv[key] = value
    return rv


class Pipeline(object):
    """
    Pipeline class for processing DyNe Pipelines

    This class implements the control framework for setting up and
    parsing complex pipelines

    Parameters
    ----------
        pipe_defs_json: str [JSON File]
            JSON file defining the pipes that will be instantiated

        pipeline_def_json: str [JSON File]
            JSON file defining the pipeline architecture that will be executed
    """

    def __init__(self, pipe_defs_json, pipeline_def_json):
        # Standard param checks
        errors.check_type(pipe_defs_json, str)
        errors.check_type(pipeline_def_json, str)
        errors.check_path(pipe_defs_json, exist=True)
        errors.check_path(pipeline_def_json, exist=True)

        # Read in the JSON files
        pipe_defs = json.load(open(pipe_defs_json, 'r'),
                              object_hook=_decode_dict)
        pipeline_def = json.load(open(pipeline_def_json, 'r'),
                                 object_hook=_decode_dict)

        # Initialize the pipe defs
        self.pipes = {}
        for pipe in pipe_defs:
            # Load module
            module = importlib.import_module(pipe['PIPE_MODULE'])
            cls = getattr(module, pipe['PIPE_CLASS'])
            inst = cls(**pipe['PIPE_PARAM'])
            self.pipes[pipe['PIPE_NAME']] = inst
        self.pipes['None'] = None

        # Parse the pipeline def into a pipeline
        for us_pipe, value in pipeline_def.iteritems():

            print('{} ----> {}'.format(us_pipe, value))
            self.pipes[us_pipe].link(
                map(lambda k: self.pipes[k], value))

    def start_pipeline(self):
        self.pipes['SOURCE'].apply_pipe_as_source()
