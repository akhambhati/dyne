.. _install:

============
Installation
============
This section explains how to install DyNe.

Python is prerequisite for running DyNe. To install DyNe, please select one of
the following scenarios:

    - :ref:`Installing Python <new-to-python>`
    - :ref:`Installing DyNe <new-to-dyne>`
    - :ref:`Upgrading a DyNe installation <upgrade-dyne>`

.. note::

     All installation commands should be run in a Terminal (for Mac and Linux).
     DyNe has not yet been tested on Windows.


.. _new-to-python:

Installing Python
--------------------------
For new and experienced users, we **highly recommend** `installing Anaconda
<https://www.continuum.io/downloads>`_. Anaconda conveniently
installs Python and other commonly used packages for scientific computing and
data science. Follow Anaconda's instructions for
downloading and installing the Python 2.7.11 version.

Once Anaconda is installed,
proceed to the :ref:`next steps <new-to-dyne>` for installing DyNe.


.. _new-to-dyne:

Installing DyNe (I have Python)
---------------------------------------
.. important::
    **Prerequisite: DyNe installation requires Python 2.7.11**

Easy Way (Using Conda)
^^^^^^^^^^^^^^^^^^^^^^
If Anaconda is installed, run the following command in the Terminal
(Mac/Linux) or CommandPrompt (Windows) to install Jupyter::

    conda create -n dyne python=2.7.11
    source activate dyne
    conda install dyne

Advanced Way (Manual Install)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If Anaconda is installed and you would like access to DyNe source code,
follow the following steps::

    git clone --recursive https://github.com/akhambhati/dyne.git
    cd dyne
    conda env create -f environment.yml
    source activate dyne

External Packages
^^^^^^^^^^^^^^^^^
For most pipes, batteries are included. However a few pipes require
installing packages manually. Please see the list below:
    - adjacency.coherence.MTCoh -- `mtspec package <http://krischer.github.io/mtspec/>`_.


.. _upgrade-dyne:

Upgrading a DyNe installation
-----------------------------
**If using Anaconda**::

    conda update dyne
