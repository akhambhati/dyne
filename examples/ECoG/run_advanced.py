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

    ["pipes.surrogate.fourier.PhaseShuffle",
     {"cache": False,
      "n_total_surrogate": 100,
      "pipe_name": "Phase_Randomize"}],

    ["pipes.adjacency.correlation.Corr",
     {"cache": True,
      "pipe_name": 'PearsonCorr'}],

    ["pipes.netviz.disp.GraphDisplay",
     {"cache": False,
      "adj_lim": [(0.00, 0.33),
                  (0.33, 0.66),
                  (0.66, 1.00)],
      "sensor_map_path": options['dataset_name']+'-Map.png',
      "sensor_coords_path": options['dataset_name']+'-Coords.csv',
      "pipe_name": "NetworkStrength"}],
]

if __name__ == '__main__':
    reg_pipeline = dyne.pipeline.Pipeline(pipes=pipeline_def, opt=options)
    reg_pipeline.start()
