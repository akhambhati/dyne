"""
Installation of DyNe in Python Environment

Created By: Ankit Khambhati

Change Log
----------
2016/03/12 - Implemented setup.py module
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open('./README.md', 'r') as rmf:
    readme = rmf.read()

# versioning
import dyne
setup(
    name = 'dyne',
    version = dyne.__version__,
    author = 'Ankit Khambhati',
    author_email = 'akhambhati@gmail.com',
    url = 'https://github.com/akhambhati/dyne',
    license = 'BSD-3',
    description = 'Dynamic network toolbox for real-time' +
                  ' and offline analysis of multivariate neural signals',
    long_description = readme,
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Data Scientists',
        'Topic :: Network Neuroscience :: Analysis Tools',
        'Topic :: Complex Networks :: Analysis Tools',
        'License :: OSI Approved :: BSD 3-Clause License',
        'Programming Language :: Python :: 2.7.11',
    ],
    keywords = 'network real-time signal fmri ecog eeg neuron adjacency' +
               ' correlation coherence',
    packages=find_packages(exclude=['docs', 'tests']),
    data_file = [('', ['LICENSE.txt']),
                 ('', ['README.md'])],
    install_requires = ['python==2.7.11']
)
