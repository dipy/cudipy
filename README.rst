GPU-accelerated functions for DIPY
==================================

cuDIPY is a Python library providing GPU-based implementations of a subset of
algorithms from the DIPY software library.

This repository is at an early stage of development and should be considered
experimental. Longer-term, we plan to incorporate GPU support into DIPY itself
rather than maintaining two libraries in parallel.

Documentation
=============

The behavior of functions in this repository should match those in the
main DIPY respository. See DIPY's Documentation_ for details of individual
functions.

.. _main repository: http://github.com/dipy/cudipy
.. _Documentation: http://dipy.org


Requirements
============
The following requirements should be installed prior to installing cuDIPY.

numpy >= 1.15
cupy >= 8.0
cupyimg
nibabel >= 3.0.0
scipy >= 1.0

cupyimg currently is not on PyPI and should be installed after CuPy directly
from its repository via::

    pip install git+https://github.com/mritools/cupyimg


Installing cuDIPY
=================

cuDIPY is not currently on PyPI, but can be installed from the repository
using `pip`::

    pip install git+https://github.com/dipy/cudipy.git


Available Functionality
=======================

This library implements what is currently a relatively small, but useful subset
of DIPY. Currently this include some the following primary functionality:

- Non-rigid registration via `cudipy.align.SymmetricDiffeomorphicRegistration`
  (SyN) using a normalized cross-correlation metric (CCMetric). The AffineMap,
  DiffeoMorphicMap and ScaleSpace classes have also been implemented.

- Tissue segmentation using `cudipy.segment.TissueClassifierHMRF`.

- Gibbs artifactd removal via `cudipy.denoise.gibbs_removal`

- Noise estimation via `cudipy.denoise.noise_estimate` and
  `cudipy.denoise.pca_noise_estimate`.

- Brain extraction via ``median_otsu``

In general, the functions and classes operate in the same way as their DIPY
counterparts, but take CuPy arrays as inputs rather than NumPy arrays. The
functions have only been tested on NVIDIA GPUs, but some functionality may also
work on AMD GPUs via CuPy's initial HIP/ROCm support.
