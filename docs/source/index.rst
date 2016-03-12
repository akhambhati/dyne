.. _landing:

==========================================================
Dynamic Network (DyNe) Analysis Toolbox for Neural Signals
==========================================================

DyNe is a pipelined toolbox to analyze streaming multivariate neural signals
in real-time or offline applications

.. sidebar:: What's new in DyNe
    **Release 0.5**

    - State-of-the-art pre-processing tools for ECoG-based functional networks
    - Pipeline construction and formatting in accessible JSON format
    - Real-time caching to HDF5 file-format for offline analysis
    - Novel pipeline logging system to track experimental runs and parameters

    To upgrade to the release:
        ``conda upgrade dyne``

Using DyNe, you can construct complex pipelines to construct, analyze and
visualize dynamic networks from multivariate neural signals. DyNe emerges from
the rapidly growing field of network neuroscience as a toolbox for mapping
high-dimensional systems with time-varying architecture. We built DyNe to
address the following critical needs in network neuroscience:

    - Standardization of scientific methods
    - Reproducibility of scientific findings
    - Extensibility for new algorithms as the field evolves
    - Prototyping of analysis where many different pre-processing and analysis
      steps must be considered

.. toctree::
    :maxdepth: 2
    :caption: User Documentation

    install
