"""
Pipeline control framework for instantiating, logging, and resuming
complex pipelines

Created by: Ankit Khambhati

Change Log
----------
2016/03/08 - Established the BasePipe
"""

import pandas as pd
import json
import h5py
import importlib
import time

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
        # check pipe_defs has right content
        errors.check_type(pipe_defs, dict)
        errors.check_has_key(pipe_defs, 'SOURCE')
        errors.check_has_key(pipe_defs, 'FLOW')
        errors.check_has_key(pipe_defs, 'LOG')
        errors.check_type(pipe_defs['SOURCE'], dict)
        errors.check_type(pipe_defs['FLOW'], list)
        errors.check_type(pipe_defs['LOG'], dict)
        errors.check_path(pipe_defs['LOG']['PATH'], exist=False)

        # Combine all pipes for initialization
        all_pipes = [pipe_defs['SOURCE']]
        for p in pipe_defs['FLOW']:
            all_pipes.append(p)

        # Instantiate each pipe
        self.pipes = {}
        for pipe in all_pipes:
            # Load module
            module = importlib.import_module(pipe['PIPE_MODULE'])
            cls = getattr(module, pipe['PIPE_CLASS'])
            inst = cls(**pipe['PIPE_PARAM'])
            self.pipes[pipe['PIPE_NAME']] = inst
        self.pipes['None'] = None

        # Store name of the source pipe
        self.pipes_srcname = pipe_defs['SOURCE']['PIPE_NAME']

        # Parse the pipeline def into a pipeline
        for us_pipe, value in pipeline_def.iteritems():
            upstream_inst = self.pipes[us_pipe]
            upstream_inst.link(
                map(lambda k: self.pipes[k], value))

        # Generate a log cross-referencing the pipe_name, pipe_class,
        # source and target hash ids, and the linking date
        log_entries = []
        link_date = time.strftime('%Y-%m-%d %H:%M:%S',
                                  time.localtime())
        for us_pipe, value in pipeline_def.iteritems():
            upstream_inst = self.pipes[us_pipe]

            for v in value:
                downstream_inst = self.pipes[v]
                if downstream_inst is not None:
                    log = {
                        'DATE': link_date,
                        'PIPE_NAME': v,
                        'PIPE_CLASS': downstream_inst.__module__ + \
                                      downstream_inst.__class__.__name__,
                        'PIPE_PARAM': downstream_inst.to_JSON(),
                        'UPSTREAM_HASH': upstream_inst.to_hash(),
                        'DOWNSTREAM_HASH': downstream_inst.to_hash()}
                    log_entries.append(log)
        self.log_entries_df = pd.DataFrame(log_entries)
        self.log_entries_path = pipe_defs['LOG']['PATH']

    def start_pipeline(self):
        self.pipes[self.pipes_srcname].apply_pipe_as_source()
        self.log_entries_df.to_csv(self.log_entries_path)
