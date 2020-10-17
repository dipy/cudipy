import cupy as cp
import numpy as np

from cudipy._utils import get_array_module


def _add_gaussian(sig, noise1, noise2):
    """
    Helper function to add_noise

    This one simply adds one of the Gaussians to the sig and ignores the other
    one.
    """
    return sig + noise1


@cp.fuse(kernel_name='_add_rician')
def _add_rician_fused(sig, noise1, noise2):
    """
    Helper function to add_noise.

    This does the same as abs(sig + complex(noise1, noise2))

    """
    tmp = (sig + noise1)
    tmp *= tmp
    return cp.sqrt(tmp + noise2 * noise2)


def _add_rician_numpy(sig, noise1, noise2):
    """
    Helper function to add_noise.

    This does the same as abs(sig + complex(noise1, noise2))

    """
    tmp = (sig + noise1)
    tmp *= tmp
    return np.sqrt(tmp + noise2 * noise2)


def _add_rician(sig, noise1, noise2):
    xp = get_array_module(sig, noise1, noise2)
    if xp == cp:
        # use fused kernel for optimal performance
        return _add_rician_fused(sig, noise1, noise2)
    else:
        return _add_rician_numpy(sig, noise1, noise2)


@cp.fuse(kernel_name='_add_rayleigh')
def _add_rayleigh_fused(sig, noise1, noise2):
    r"""Helper function to add_noise.

    The Rayleigh distribution is $\sqrt\{Gauss_1^2 + Gauss_2^2}$.

    """
    return sig + cp.sqrt(noise1 * noise1 + noise2 * noise2)


def _add_rayleigh_numpy(sig, noise1, noise2):
    r"""Helper function to add_noise.

    The Rayleigh distribution is $\sqrt\{Gauss_1^2 + Gauss_2^2}$.

    """
    return sig + np.sqrt(noise1 * noise1 + noise2 * noise2)


def _add_rayleigh(sig, noise1, noise2):
    xp = get_array_module(sig, noise1, noise2)
    if xp == cp:
        # use fused kernel for optimal performance
        return _add_rayleigh_fused(sig, noise1, noise2)
    else:
        return _add_rayleigh_numpy(sig, noise1, noise2)


def add_noise(signal, snr, S0, noise_type="rician"):
    r""" Add noise of specified distribution to the signal from a single voxel.

    Parameters
    -----------
    signal : 1-d ndarray
        The signal in the voxel.
    snr : float
        The desired signal-to-noise ratio. (See notes below.)
        If `snr` is None, return the signal as-is.
    S0 : float
        Reference signal for specifying `snr`.
    noise_type : string, optional
        The distribution of noise added. Can be either 'gaussian' for Gaussian
        distributed noise, 'rician' for Rice-distributed noise (default) or
        'rayleigh' for a Rayleigh distribution.

    Returns
    --------
    signal : array, same shape as the input
        Signal with added noise.

    Notes
    -----
    SNR is defined here, following [1]_, as ``S0 / sigma``, where ``sigma`` is
    the standard deviation of the two Gaussian distributions forming the real
    and imaginary components of the Rician noise distribution (see [2]_).

    References
    ----------
    .. [1] Descoteaux, Angelino, Fitzgibbons and Deriche (2007) Regularized,
           fast and robust q-ball imaging. MRM, 58: 497-510
    .. [2] Gudbjartson and Patz (2008). The Rician distribution of noisy MRI
           data. MRM 34: 910-914.

    Examples
    --------
    >>> signal = np.arange(800).reshape(2, 2, 2, 100)
    >>> signal_w_noise = add_noise(signal, 10., 100., noise_type='rician')

    """
    if snr is None:
        return signal
    xp = get_array_module(signal)

    sigma = S0 / snr

    noise_adder = {
        "gaussian": _add_gaussian,
        "rician": _add_rician,
        "rayleigh": _add_rayleigh,
    }

    # ensure single precision output for single precision inputs
    float_dtype = xp.promote_types(signal.dtype, np.float32)

    noise1 = xp.random.normal(0, sigma, size=signal.shape)
    noise1 = noise1.astype(float_dtype, copy=False)

    if noise_type == "gaussian":
        noise2 = None
    else:
        noise2 = xp.random.normal(0, sigma, size=signal.shape)
        noise2 = noise2.astype(float_dtype, copy=False)

    return noise_adder[noise_type](signal, noise1, noise2)
