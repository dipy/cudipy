import numpy as np

import cupy as cp
import cupyx.scipy.linalg

import cupyx.scipy.ndimage as ndi
import cupyx.scipy.special as sps


def pca_noise_estimate(
    data,
    gtab,
    patch_radius=1,
    correct_bias=True,
    smooth=2,
    *,
    allow_single=False,
):
    """ PCA based local noise estimation.

    Parameters
    ----------
    data: 4D array
        the input dMRI data.

    gtab: gradient table object
      gradient information for the data gives us the bvals and bvecs of
      diffusion data, which is needed here to select between the noise
      estimation methods.
    patch_radius : int
        The radius of the local patch to be taken around each voxel (in
        voxels). Default: 1 (estimate noise in blocks of 3x3x3 voxels).
    correct_bias : bool
      Whether to correct for bias due to Rician noise. This is an implementation
      of equation 8 in [1]_.

    smooth : int
      Radius of a Gaussian smoothing filter to apply to the noise estimate
      before returning. Default: 2.

    Returns
    -------
    sigma_corr: 3D array
        The local noise standard deviation estimate.

    References
    ----------
    .. [1] Manjon JV, Coupe P, Concha L, Buades A, Collins DL "Diffusion
           Weighted Image Denoising Using Overcomplete Local PCA". PLoS ONE
           8(9): e73021. doi:10.1371/journal.pone.0073021.
    """
    # first identify the number of the b0 images
    K = np.count_nonzero(gtab.b0s_mask)

    if K > 1:
        # If multiple b0 values then use MUBE noise estimate
        data0 = data[..., cp.asarray(gtab.b0s_mask)]
        # sibe = False

    else:
        # if only one b0 value then SIBE noise estimate
        data0 = data[..., cp.asarray(~gtab.b0s_mask)]
        # sibe = True

    n0, n1, n2, n3 = data0.shape
    nsamples = n0 * n1 * n2

    if allow_single:
        data_dtype = cp.promote_types(data0.dtype, cp.float32)
    else:
        data_dtype = cp.float64
    data0 = data0.astype(data_dtype, copy=False)
    X = data0.reshape(nsamples, n3)
    # Demean:
    X = X - X.mean(axis=0, keepdims=True)
    # compute the covariance matrix, x
    r = cp.dot(X.T, X)
    # (symmetric) eigen decomposition
    w, v = cp.linalg.eigh(r)
    # project smallest eigenvector/value onto the data space
    I = X.dot(v[:, 0:1]).reshape(n0, n1, n2)
    del r, w, v

    s = 2 * patch_radius + 1
    sum_reg = ndi.uniform_filter(I, size=s)
    sigma_sq = I - sum_reg
    sigma_sq *= sigma_sq

    # find the SNR and make the correction for bias due to Rician noise:
    if correct_bias:
        mean = ndi.uniform_filter(data0.mean(-1), size=s, mode="reflect")
        snr = mean / cp.sqrt(sigma_sq)
        snr_sq = snr * snr
        # snr_sq = cp.asnumpy(snr_sq)  # transfer to host to use sps.iv
        # xi is practically equal to 1 above 37.4, and we overflow, raising
        # warnings and creating ot-a-numbers.
        # Instead, we will replace these values with 1 below
        with np.errstate(over="ignore", invalid="ignore"):
            tmp1 = snr_sq / 4
            tmp = sps.i0(tmp1)
            tmp *= 2 + snr_sq
            tmp += snr_sq * sps.i1(tmp1)
            tmp *= tmp
            tmp *= (np.pi / 8) * cp.exp(-snr_sq / 2)
            xi = 2 + snr_sq - tmp
            xi = xi.astype(data_dtype, copy=False)
            # xi = (2 + snr_sq - (np.pi / 8) * cp.exp(-snr_sq / 2) *
            #       ((2 + snr_sq) * sps.i0(snr_sq / 4) +
            #       (snr_sq) * sps.i1(snr_sq / 4)) ** 2).astype(float)
        xi[snr > 37.4] = 1
        sigma_corr = sigma_sq / xi
        sigma_corr[cp.isnan(sigma_corr)] = 0
    else:
        sigma_corr = sigma_sq

    if smooth is not None:
        ndi.gaussian_filter(sigma_corr, smooth, output=sigma_corr)

    cp.sqrt(sigma_corr, out=sigma_corr)
    return sigma_corr
