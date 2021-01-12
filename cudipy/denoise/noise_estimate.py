import math

import numpy as np

import cupy as cp
from cupyx.scipy import ndimage as ndi


def estimate_sigma(arr, disable_background_masking=False, N=0):
    """Standard deviation estimation from local patches

    Parameters
    ----------
    arr : 3D or 4D ndarray
        The array to be estimated

    disable_background_masking : bool, default False
        If True, uses all voxels for the estimation, otherwise, only non-zeros
        voxels are used. Useful if the background is masked by the scanner.

    N : int, default 0
        Number of coils of the receiver array. Use N = 1 in case of a SENSE
        reconstruction (Philips scanners) or the number of coils for a GRAPPA
        reconstruction (Siemens and GE). Use 0 to disable the correction factor,
        as for example if the noise is Gaussian distributed. See [1] for more
        information.

    Returns
    -------
    sigma : ndarray
        standard deviation of the noise, one estimation per volume.

    Notes
    -------
    This function is the same as manually taking the standard deviation of the
    background and gives one value for the whole 3D array.
    It also includes the coil-dependent correction factor of Koay 2006
    (see [1]_, equation 18) with theta = 0.
    Since this function was introduced in [2]_ for T1 imaging,
    it is expected to perform ok on diffusion MRI data, but might oversmooth
    some regions and leave others un-denoised for spatially varying noise
    profiles. Consider using :func:`piesno` to estimate sigma instead if visual
    inaccuracies are apparent in the denoised result.

    References
    ----------
    .. [1] Koay, C. G., & Basser, P. J. (2006). Analytically exact correction
    scheme for signal extraction from noisy magnitude MR signals.
    Journal of Magnetic Resonance), 179(2), 317-22.

    .. [2] Coupe, P., Yger, P., Prima, S., Hellier, P., Kervrann, C., Barillot,
    C., 2008. An optimized blockwise nonlocal means denoising filter for 3-D
    magnetic resonance images, IEEE Trans. Med. Imaging 27, 425-41.

    """
    k = np.zeros((3, 3, 3), dtype=np.int8)

    k[0, 1, 1] = 1
    k[2, 1, 1] = 1
    k[1, 0, 1] = 1
    k[1, 2, 1] = 1
    k[1, 1, 0] = 1
    k[1, 1, 2] = 1
    k = cp.asarray(k)

    # Precomputed factor from Koay 2006, this corrects the bias of magnitude
    # image
    correction_factor = {
        0: 1,  # No correction
        1: 0.42920367320510366,
        4: 0.4834941393603609,
        6: 0.4891759468548269,
        8: 0.49195420135894175,
        12: 0.4946862482541263,
        16: 0.4960339908122364,
        20: 0.4968365823718557,
        24: 0.49736907650825657,
        32: 0.49803177052530145,
        64: 0.49901964176235936,
    }

    if N in correction_factor:
        factor = correction_factor[N]
    else:
        raise ValueError(
            "N = {0} is not supported! Please choose amongst \
{1}".format(
                N, sorted(list(correction_factor.keys()))
            )
        )

    if arr.ndim == 3:
        arr = arr[..., None]
    elif arr.ndim != 4:
        raise ValueError("Array shape is not supported!", arr.shape)

    if disable_background_masking:
        mask = None
    else:
        mask = arr[..., 0].astype(np.bool)
        # TODO: make upstream PR at dipy with this binary erosion bug fix
        # erode mask by the convolution kernel shape
        mask = ndi.binary_erosion(
            mask, structure=cp.ones(k.shape[:3], dtype=np.bool)
        )

    # TODO: make upstream PR at dipy that avoids an explicit loop over slices
    conv_out = cp.empty(arr.shape, dtype=np.float64)

    ndi.convolve(arr, k[..., np.newaxis], output=conv_out)
    mean_block = arr - conv_out / 6
    if mask is None:
        tmp = mean_block.reshape((-1, mean_block.shape[-1]))
    else:
        tmp = mean_block[mask]
    tmp *= math.sqrt(6 / 7)
    tmp *= tmp
    sigma = cp.sqrt(cp.mean(tmp, axis=0) / factor)
    return sigma
