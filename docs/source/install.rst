.. _install:

============
Installation
============

This section explains how to install DyNe.

Python is prerequisite for running DyNe. To install DyNe, please select one of
the following scenarios:

    - :ref:`Installing DyNe and Python <new-to-python-and-dyne>`
    - :ref:`Installing DyNe (I already have Python) <existing-python-new-dyne>`
    - :ref:`Upgrading a DyNe installation <upgrading>`

.. note::

     All installation commands should be run in a Terminal (for Mac and Linux).
     DyNe has not yet been tested on Windows.


.. _new-to-python-and-dyne:

Installing DyNe and Python
--------------------------

For new and experienced users, we **highly recommend** `installing Anaconda
<https://www.continuum.io/downloads>`_. Anaconda conveniently
installs Python and other commonly used packages for scientific computing and
data science. Follow Anaconda's instructions for
downloading and installing the Python 2.7 version.

Once Anaconda is installed,
proceed to the :ref:`next steps <existing-python-new-dyne>` for installing DyNe.


.. _existing-python-new-dyne:

Installing DyNe (I already have Python)
---------------------------------------

.. important::

    **Prerequisite: DyNe installation requires Python 2.7**

Using Anaconda and conda
^^^^^^^^^^^^^^^^^^^^^^^^

If Anaconda is installed, run the following command in the Terminal
(Mac/Linux) or CommandPrompt (Windows) to install Jupyter::

    conda install dyne2

.. note::

    Some of DyNe's dependencies may require compilation,
    in which case you would need the ability to compile Python C-extensions.
    This means a C compiler and the Python headers.
    On Debian-based systems (e.g. Ubuntu), you can get this with::

        apt-get install build-essential python-dev

    And on Fedora-based systems (e.g. Red Hat, CentOS)::

        yum groupinstall 'Development Tools'
        yum install python-devel


.. _upgrading:

Upgrading a DyNe installation
-----------------------------

**If using Anaconda**::

    conda update dyne2
