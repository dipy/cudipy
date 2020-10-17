import numpy as np
import cupy as cp


d = cp.cuda.Device()


def _sl_axis(axis, ndim, slc):
    sl = [slice(None),] * ndim
    sl[axis] = slc
    return tuple(sl)


def _image_tv(x, h, axis=0, *, xp=None):
    """ Computes total variation (TV) of matrix x across a given axis and
    along two directions.

    Parameters
    ----------
    x : 2D ndarray
        matrix x
    h : int
        Preinitialized array of ones. This corresponds to the number of points
        to be included in TV calculation.
    axis : int (0 or 1)
        Axis which TV will be calculated. Default a is set to 0.

    Returns
    -------
    ptv : 2D ndarray
        Total variation calculated from the right neighbours of each point.
    ntv : 2D ndarray
        Total variation calculated from the left neighbours of each point.

    """
    if xp is None:
        xp = cp.get_array_module(x)
    if xp == np:
        import scipy.ndimage as ndi
    else:
        import cupyx.scipy.ndimage as ndi
    ndim = x.ndim

    n_points = h.size

    # Add copies of the data so that data extreme points are also analysed
    pad_width = [(0, 0),] * ndim
    pad_width[axis] = (n_points + 1, n_points + 1)
    xs = xp.pad(x, pad_width=pad_width, mode="wrap")

    diff_sl1 = _sl_axis(axis, ndim, slice(1, -1))
    diff_sl_p = _sl_axis(axis, ndim, slice(2, None))
    diff_sl_n = _sl_axis(axis, ndim, slice(0, -2))
    tmp = xs[diff_sl1]
    pdiff = xp.abs(tmp - xs[diff_sl_p])
    ndiff = xp.abs(tmp - xs[diff_sl_n])

    center_sl = _sl_axis(axis, ndim, slice(n_points, -n_points))
    ptv = ndi.convolve1d(
        pdiff, h, origin=(n_points - 1) // 2, axis=axis)[center_sl]
    ntv = ndi.convolve1d(
        ndiff, h, origin=-(n_points // 2), axis=axis)[center_sl]
    return ptv, ntv


def _gibbs_removal_1d(x, axis=0, n_points=3, xp=None):
    """Suppresses Gibbs ringing along a given axis using fourier sub-shifts.

    Parameters
    ----------
    x : 2D ndarray
        Matrix x.
    axis : int (0 or 1)
        Axis in which Gibbs oscillations will be suppressed.
        Default is set to 0.
    n_points : int, optional
        Number of neighbours to access local TV (see note).
        Default is set to 3.

    Returns
    -------
    xc : 2D ndarray
        Matrix with suppressed Gibbs oscillations along the given axis.

    Notes
    -----
    This function suppresses the effects of Gibbs oscillations based on the
    analysis of local total variation (TV). Although artefact correction is
    done based on two adjacent points for each voxel, total variation should be
    accessed in a larger range of neighbours. The number of neighbours to be
    considered in TV calculation can be adjusted using the parameter n_points.

    """
    if xp is None:
        xp = cp.get_array_module(x)
    float_dtype = xp.promote_types(x.dtype, np.float32)
    ssamp = xp.linspace(0.02, 0.9, num=45, dtype=float_dtype)

    xs = xp.moveaxis(x, axis, -1).copy()
    h = xp.ones(n_points, dtype=x.real.dtype)  # filter used in _image_tv

    # TV for shift zero (baseline)
    tvr, tvl = _image_tv(xs, h, axis=-1)
    tvp = xp.minimum(tvr, tvl)
    tvn = tvp.copy()

    # Find optimal shift for gibbs removal
    isp = xs.copy()
    isn = xs.copy()
    sp = xp.zeros(xs.shape, dtype=float_dtype)
    sn = xp.zeros(xs.shape, dtype=float_dtype)
    n = xs.shape[-1]
    c = xp.fft.fft(xs, axis=-1)
    k = xp.fft.fftfreq(n, 1 / (2.0j * np.pi))
    k = k.astype(xp.promote_types(xs.dtype, xp.complex64), copy=False)
    if xs.ndim == 2:
        k = k[np.newaxis, :]
        ssamp_nd = ssamp[:, np.newaxis]
    elif xs.ndim == 3:
        k = k[np.newaxis, np.newaxis, :]
        ssamp_nd = ssamp[:, np.newaxis, np.newaxis]
    all_eks = ssamp_nd * k
    xp.exp(all_eks, out=all_eks)
    for s, eks in zip(ssamp, all_eks):
        eks = eks[np.newaxis, ...]
        # Access positive shift for given s
        img_p = c * eks
        img_p = xp.fft.ifft(img_p, axis=-1)
        xp.abs(img_p, out=img_p)

        tvsr, tvsl = _image_tv(img_p, h, axis=-1)
        tvs_p = xp.minimum(tvsr, tvsl)

        # Access negative shift for given s
        img_n = c * xp.conj(eks)  # xp.exp(-ks)
        img_n = xp.fft.ifft(img_n, axis=-1)
        xp.abs(img_n, out=img_n)

        tvsr, tvsl = _image_tv(img_n, h, axis=-1)
        tvs_n = xp.minimum(tvsr, tvsl)

        maskp = tvp > tvs_p
        maskn = tvn > tvs_n

        # Update positive shift params
        isp[maskp] = img_p[maskp].real
        sp[maskp] = s
        tvp[maskp] = tvs_p[maskp]

        # Update negative shift params
        isn[maskn] = img_n[maskn].real
        sn[maskn] = s
        tvn[maskn] = tvs_n[maskn]

    # check non-zero sub-voxel shifts
    idx = xp.nonzero(sp + sn)

    # use positive and negative optimal sub-voxel shifts to interpolate to
    # original grid points
    sn_i = sn[idx]
    isn_i = isn[idx]
    tmp = isp[idx] - isn_i
    tmp /= sp[idx] + sn_i
    tmp *= sn_i
    tmp += isn_i
    xs[idx] = tmp

    return xp.moveaxis(xs, -1, axis)


def _weights(shape, image_dtype, xp):
    """ Computes the weights necessary to combine two images processed by
    the 1D Gibbs removal procedure along two different axes [1]_.

    Parameters
    ----------
    shape : tuple
        shape of the image.

    Returns
    -------
    G0 : 2D ndarray
        Weights for the image corrected along axis 0.
    G1 : 2D ndarray
        Weights for the image corrected along axis 1.

    References
    ----------
    .. [1] Kellner E, Dhital B, Kiselev VG, Reisert M. Gibbs-ringing artifact
           removal based on local subvoxel-shifts. Magn Reson Med. 2016
           doi: 10.1002/mrm.26054.

    """
    dtype = np.promote_types(np.float32, image_dtype)
    G0 = xp.zeros(shape, dtype=dtype)
    G1 = xp.zeros(shape, dtype=dtype)
    k0 = xp.linspace(-xp.pi, xp.pi, num=shape[0], dtype=dtype)
    k1 = xp.linspace(-xp.pi, xp.pi, num=shape[1], dtype=dtype)

    # Middle points
    K1, K0 = xp.meshgrid(k1[1:-1], k0[1:-1], sparse=True)
    cosk0 = 1.0 + xp.cos(K0)
    cosk1 = 1.0 + xp.cos(K1)
    denom = cosk0 + cosk1
    G1[1:-1, 1:-1] = cosk0 / denom
    G0[1:-1, 1:-1] = cosk1 / denom

    # Boundaries
    G1[1:-1, 0] = G1[1:-1, -1] = 1
    G1[0, 0] = G1[-1, -1] = G1[0, -1] = G1[-1, 0] = 0.5
    G0[0, 1:-1] = G0[-1, 1:-1] = 1
    G0[0, 0] = G0[-1, -1] = G0[0, -1] = G0[-1, 0] = 0.5

    return G0, G1


def _gibbs_removal_2d_or_3d(image, n_points=3, G0=None, G1=None, *, xp=None):
    """ Suppress Gibbs ringing of a 2D image.

    Parameters
    ----------
    image : 2D ndarray
        Matrix containing the 2D image.
    n_points : int, optional
        Number of neighbours to access local TV (see note). Default is
        set to 3.
    G0 : 2D ndarray, optional.
        Weights for the image corrected along axis 0. If not given, the
        function estimates them using the function :func:`_weights`.
    G1 : 2D ndarray
        Weights for the image corrected along axis 1. If not given, the
        function estimates them using the function :func:`_weights`.

    Returns
    -------
    imagec : 2D ndarray
        Matrix with Gibbs oscillations reduced along axis a.

    Notes
    -----
    This function suppresses the effects of Gibbs oscillations based on the
    analysis of local total variation (TV). Although artefact correction is
    done based on two adjacent points for each voxel, total variation should be
    accessed in a larger range of neighbours. The number of neighbours to be
    considered in TV calculation can be adjusted using the parameter n_points.

    References
    ----------
    Please cite the following articles
    .. [1] Neto Henriques, R., 2018. Advanced Methods for Diffusion MRI Data
           Analysis and their Application to the Healthy Ageing Brain
           (Doctoral thesis). https://doi.org/10.17863/CAM.29356
    .. [2] Kellner E, Dhital B, Kiselev VG, Reisert M. Gibbs-ringing artifact
           removal based on local subvoxel-shifts. Magn Reson Med. 2016
           doi: 10.1002/mrm.26054.

    """
    if xp is None:
        xp = cp.get_array_module(image)
    if G0 is None or G1 is None:
        G0, G1 = _weights(image.shape[:2], image.dtype, xp=xp)
        if image.ndim > 2:
            G0 = G0[..., np.newaxis]
            G1 = G1[..., np.newaxis]

    if image.ndim not in [2, 3]:
        raise ValueError(
            "expected a 2D image or a 3D array corresponding to a batch of 2D "
            "images stacked along the last axis"
        )
    img_c1 = _gibbs_removal_1d(image, axis=1, n_points=n_points)
    img_c0 = _gibbs_removal_1d(image, axis=0, n_points=n_points)

    C1 = xp.fft.fftn(img_c1, axes=(0, 1))
    C0 = xp.fft.fftn(img_c0, axes=(0, 1))
    imagec = xp.fft.fftshift(C1, axes=(0, 1)) * G1
    imagec += xp.fft.fftshift(C0, axes=(0, 1)) * G0
    imagec = xp.fft.ifftn(imagec, axes=(0, 1))
    imagec = xp.abs(imagec)
    return imagec


def gibbs_removal(vol, slice_axis=2, n_points=3, inplace=False,
                  num_threads=None, *, xp=None):
    """Suppresses Gibbs ringing artefacts of images volumes.

    Parameters
    ----------
    vol : ndarray ([X, Y]), ([X, Y, Z]) or ([X, Y, Z, g])
        Matrix containing one volume (3D) or multiple (4D) volumes of images.
    slice_axis : int (0, 1, or 2)
        Data axis corresponding to the number of acquired slices.
        Default is set to the third axis.
    n_points : int, optional
        Number of neighbour points to access local TV (see note).
        Default is set to 3.
    inplace : bool, optional
        unimplemented option on the GPU
    num_threads : int or None, optional
        unsupported option on the GPU

    Returns
    -------
    vol : ndarray ([X, Y]), ([X, Y, Z]) or ([X, Y, Z, g])
        Matrix containing one volume (3D) or multiple (4D) volumes of corrected
        images.

    Notes
    -----
    For 4D matrix last element should always correspond to the number of
    diffusion gradient directions.

    References
    ----------
    Please cite the following articles
    .. [1] Neto Henriques, R., 2018. Advanced Methods for Diffusion MRI Data
           Analysis and their Application to the Healthy Ageing Brain
           (Doctoral thesis). https://doi.org/10.17863/CAM.29356
    .. [2] Kellner E, Dhital B, Kiselev VG, Reisert M. Gibbs-ringing artifact
           removal based on local subvoxel-shifts. Magn Reson Med. 2016
           doi: 10.1002/mrm.26054.

    """
    nd = vol.ndim

    if xp is None:
        xp = cp.get_array_module(vol)
    if xp is np:
        # The implementation here was refactored for the GPU
        # Dipy's version is faster on the CPU, so fall back to it in that case.
        from dipy.denoise.gibbs import gibbs_removal as gibbs_removal_cpu
        try:
            return gibbs_removal_cpu(vol, slice_axis=slice_axis,
                                     n_points=n_points, inplace=inplace,
                                     num_threads=num_threads)
        except TypeError:
            warnings.warn("inplace and num_threads arguments ignored")
            # older DIPY did not have inplace or num_threads kwargs
            return gibbs_removal_cpu(vol, slice_axis=slice_axis,
                                     n_points=n_points)

    if not isinstance(inplace, bool):
        raise TypeError("inplace must be a boolean.")

    if num_threads is not None:
        warnings.warn("num_threads is ignored by the GPU operation")

    # check the axis corresponding to different slices
    # 1) This axis cannot be larger than 2
    if slice_axis > 2:
        raise ValueError("Different slices have to be organized along" +
                         "one of the 3 first matrix dimensions")

    # 2) If this is not 2, swap axes so that different slices are ordered
    # along axis 2. Note that swapping is not required if data is already a
    # single image
    elif slice_axis < 2 and nd > 2:
        vol = xp.swapaxes(vol, slice_axis, 2)

    # check matrix dimension
    if nd == 4:
        inishap = vol.shape
        vol = vol.reshape((inishap[0], inishap[1], inishap[2] * inishap[3]))
    elif nd > 4:
        raise ValueError("Data have to be a 4D, 3D or 2D matrix")
    elif nd < 2:
        raise ValueError("Data is not an image")

    # Produce weigthing functions for 2D Gibbs removal
    shap = vol.shape
    G0, G1 = _weights(shap[:2], vol.dtype, xp=xp)

    if inplace:
        raise NotImplementedError("inplace restoration not supported")

    # Run Gibbs removal of 2D images
    if nd > 2:
        G0 = G0[..., np.newaxis]
        G1 = G1[..., np.newaxis]
    vol = _gibbs_removal_2d_or_3d(vol, n_points=n_points, G0=G0, G1=G1, xp=xp)

    # Reshape data to original format
    if nd == 4:
        vol = vol.reshape(inishap)
    if slice_axis < 2 and nd > 2:
        vol = xp.swapaxes(vol, slice_axis, 2)

    return vol
