GPU-accelerated functions for DIPY
==================================

cuDIPY is a Python library providing GPU-based implementations of algorithms
from the DIPY software library.

This repository is at an early stage of development and should be considered
experimental. Longer-term, we plan to incorporate GPU support into DIPY itself.

Documentation
=============

The behavior of functions in this repository should match those in the
main DIPY respository. See DIPY's Documentation_ for details of individual
functions.

.. _main repository: http://github.com/dipy/cudipy
.. _Documentation: http://dipy.org


Installing cuDIPY
=================

cuDIPY is not currently on PyPy, but can be installed from the repository
using `pip`::

    pip install git+https://github.com/dipy/cudipy.git


Requirements
============
NumPy >= 1.15
CuPy >= 8.0
