"""
Offline ECoG sensor data capturing a seizure
"""

import dyne


options = {
    'working_path': './working_results/',
    'model_name': 'CorrAdj',
    'dataset_name': 'Human_ECoG',
}

pipeline_def = [
    ["pipes.source.offline.MATSignal",
     {"cache": False,
      "signal_path": options['dataset_name']+'-seizure.mat',
      "info_path": options['dataset_name']+'-info.mat',
      "win_len": 1.0,
      "win_disp": 0.5,
      "pipe_name": 'MATSignalSource'}],

    ["pipes.adjacency.correlation.Corr",
     {"cache": True,
      "pipe_name": 'PearsonCorr'}],
]

if __name__ == '__main__':
    reg_pipeline = dyne.pipeline.Pipeline(pipes=pipeline_def, opt=options)
    reg_pipeline.start()
