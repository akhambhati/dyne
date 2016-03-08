==========================================================
Dynamic Network (DyNe) Analysis Toolbox for Neural Signals
==========================================================
Dynamic network toolbox for real-time, pipelined analysis and visualization of
sensor networks

What is DyNe?
-------------
DyNe extracts, analyzes, and presents information from sensor networks that
evolve with time. DyNe emerges from the rapidly-growing field of complex
networks as a toolbox for mapping multidimensional systems whose architecture
changes with time. DyNe comprises of state-of-the-art network tools to study
and visualize system dynamics in real-time.

Authors
-------
#### Ankit N. Khambhati [homepage](http://www.google.com)

**University of Pennsylvania, Department of Bioengineering**

Basic Usage
-----------
#### Pipeline Architecture
DyNe utilizes *pipelining* to analyze and visualize incoming sensor data and
results. A pipeline is an ordered list of modules or *pipes* that process
incoming data in logical steps. While pipeline construction follows general
rules dictating which pipes are capable of linking, the end-user has flexibility
in choosing which version of a pipe to implement depending on the specific task.

###### Building a Conceptual Pipeline
Suppose you wish to visualize a series of time-dependent adjacency networks from
small, contiguous windows of continuous sensor data stored offline. You might
create a pipeline as follows:
```
SOURCE --> ADJACENCY --> NETVIZ 
```
This pipe linearly connects instances of *SOURCE*, *ADJACENCY*
and *NETVIZ*-type pipes. Keep in mind that you must choose specific instances
of each pipe to interface.

#### Available Pipes
Here we detail the different pipe types currently available in DyNe, which type
of pipe they can link to, and specific pipes one can choose from.

<table>
  <tbody>
    <tr>
      <th align="center">Pipe Type</th>
      <th align="center">Link To</th>
      <th align="center">Pipe Instance -- Description</th>
    </tr>
    <tr>
      <td align="center">source</td>
      <td align="center">sigproc<br/>surrogate<br/>association<br/></td>
      <td align="left">[1] offline.MATSignal -- Stream sensor signals stored in MATLAB (MAT) format</td>
    </tr>
    <tr>
      <td align="center">sigproc</td>
      <td align="center">sigproc<br/>surrogate<br/>association<br/></td>
      <td align="left">[1] filters.EllipticFilter -- Apply Elliptic filter to signal</td>
    </tr>
     <tr>
      <td align="center">surrogate</td>
      <td align="center">association</td>
      <td align="left">[1] fourier.PhaseShuffle -- Generate surrogate signals with retained Fourier magnitude and randomized Fourier phases</td>
    </tr>
     <tr>
      <td align="center">adjacency</td>
      <td align="center">netviz</td>
      <td align="left">[1] correlation.Corr -- Generate association matrix from sensor signals using Pearson Correlation<br/>[2] correlation.XCorr -- Generate association matrix from sensor signals using Pearson Cross-Correlation<br/>[3] coherence.MTCoh -- Generate association matrix from sensor signals using multitaper coherence method</td>
 </tbody>
</table>

#### Initializing and Running a Pipeline
###### Setting Runtime Options
DyNe considers every distinct pipeline structure as a *model* that operates on a
*dataset* from which the signal originates. DyNe stores model parameters for
the dataset on which the pipeline was run in a *working directory*. These basic
run parameters are specified by a Python dictionary:
```python
options = {
    ‘working_path’: ‘path_string_here’
    ‘model_name’: ‘model_name_here’
    ‘dataset_name’: ‘dataset_name_here’
}
```
These options are cached for reference in the following JSON file:
```
working_path/model_name/dataset_name_options.json
```

###### Defining a Pipeline
Pipelines are constructed by specifying an ordered list-of-list comprised of
pipe instances and associated input parameters. Pipe instances may be found from
the table above (*see Available Pipes*), and parameters for the pipe instance
can be found in the pipe instance documentation. Pipelines are generally defined
as follows:
```python
pipeline_def = [
    ["string_containing_pipe_instance_#1",
     {"pipe_instance_param_#1": value,
      "pipe_instance_param_#2": value}],

    ["string_containing_pipe_instance_#2",
     {"pipe_instance_param_#1": value,
      "pipe_instance_param_#2": value}],

    ["string_containing_pipe_instance_#3",
     {"pipe_instance_param_#1": value,
      "pipe_instance_param_#2": value}],
]
```
The pipeline *model* is cached for reference in the following JSON file:
```
working_path/model_name/dataset_name_pipeline.json
```

###### Running a Pipeline
After setting the runtime options and pipeline definition, you can register
your settings and subsequently run your pipeline with DyNe's *Pipeline* class
as follows:
```python
import dyne

reg_pipeline = dyne.pipeline.Pipeline(pipes=pipeline_def, opt=options)
reg_pipeline.start()
```
Invoking the *start()* method will display the pipeline on-screen and begin
processing the incoming signal.

###### Basic Example Pipeline (Electrocorticography)
Let us revisit the conceptual pipeline set forth earlier for generating a
series of time-dependent adjacency matrices from small, contiguous windows of
continuous sensor data stored offline. The conceptual pipeline is defined as:
```
SOURCE --> ADJACENCY --> NETVIZ
```
In the DyNe *examples/ECoG* directory we have offline sensor data from electrocorticography stored in a MATLAB MAT-file. Examining the dataset and
conceptual pipeline we decide to specify the following runtime options:
```python
options = {
    ‘working_path’: ‘./working_results/’
    ‘model_name’: ‘CorrAdj’
    ‘dataset_name’: ‘Human_ECoG’
}
```
Secondly, we implement our conceptual pipeline as follows:
```python
pipeline_def = [
    ["pipes.source.offline.MATSignal",
     {"cache": False,
      "signal_path": options['dataset_name']+'-seizure.mat',
      "info_path": options['dataset_name']+'-info.mat',
      "win_len": 1.0,
      "win_disp": 0.5,
      "pipe_name": ‘MATSignalSource’}],

    ["pipes.adjacency.correlation.Corr",
     {"cache": True,
      "pipe_name": ‘PearsonCorr’}],

    ["pipes.netviz.disp.GraphDisplay",
     {"cache": False,
      "adj_lim": [(0.00, 0.33),
                  (0.33, 0.66),
                  (0.66, 1.00)],
      "sensor_map_path": options['dataset_name']+'-Map.png',
      "sensor_coords_path": options['dataset_name']+'-Coords.csv',
      "pipe_name": "NetworkStrength"}],
]
```
The pipeline can be run with the above parameters as detailed in the *Running a Pipeline* subsection.

This full example can be automatically run by calling
*examples/ECoG/run_basic.py* at the command-prompt.

Citing
------
#### TBD

References
----------
