"""PCA noise estimation demo extracted from (denoise_localpca.py)
"""

import cupy as cp
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from time import time
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs

from cudipy.denoise.pca_noise_estimate import pca_noise_estimate

"""

Load one of the datasets. These data were acquired with 63 gradients and 1
non-diffusion (b=0) image.

"""

dwi_fname, dwi_bval_fname, dwi_bvec_fname = get_fnames('isbi2013_2shell')
data, affine = load_nifti(dwi_fname)
bvals, bvecs = read_bvals_bvecs(dwi_bval_fname, dwi_bvec_fname)
gtab = gradient_table(bvals, bvecs)

print("Input Volume", data.shape)

"""
Estimate the noise standard deviation
=====================================

We use the ``pca_noise_estimate`` method to estimate the value of sigma to be
used in local PCA algorithm proposed by Manjon et al. [Manjon2013]_.
It takes both data and the gradient table object as input and returns an
estimate of local noise standard deviation as a 3D array. We return a smoothed
version, where a Gaussian filter with radius 3 voxels has been applied to the
estimate of the noise before returning it.

We correct for the bias due to Rician noise, based on an equation developed by
Koay and Basser [Koay2006]_.

"""

data = cp.asarray(data)

t = time()
sigma = pca_noise_estimate(data, gtab, correct_bias=True, smooth=3)
print("Sigma estimation time", time() - t)
