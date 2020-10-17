"""Demo of estimate_sigma (extracted from DIPY's denoise_nlmeans.py)
"""

import cupy as cp
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from time import time
from cudipy.denoise.noise_estimate import estimate_sigma
from dipy.data import get_fnames
from dipy.io.image import load_nifti


dwi_fname, dwi_bval_fname, dwi_bvec_fname = get_fnames('sherbrooke_3shell')
data, affine = load_nifti(dwi_fname)

mask = data[..., 0] > 80

# We select only one volume for the example to run quickly.
data = data[..., 1]

print("vol size", data.shape)

# lets create a noisy data with Gaussian data

"""
In order to call ``non_local_means`` first you need to estimate the standard
deviation of the noise. We use N=4 since the Sherbrooke dataset was acquired
on a 1.5T Siemens scanner with a 4 array head coil.
"""

data = cp.asarray(data)

tstart = time()
sigma = estimate_sigma(data, N=4)
duration = time() - tstart
print(f"duration = {duration} s")
