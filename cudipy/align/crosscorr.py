import cupy as cp
import cupyx.scipy.ndimage as ndi


_cc_precompute = cp.ElementwiseKernel(
    in_params="W ad, W bd, W sum_a, W sum_b, W sum_ab, W sum_aa, W sum_bb, int32 cnt",  # noqa
    out_params="W Ii, W Ji, W sfm, W sff, W smm",
    operation="""
    W a_mean = sum_a / cnt;
    W b_mean = sum_b / cnt;
    Ii = ad - a_mean;
    Ji = bd - b_mean;
    sfm = sum_ab - b_mean * sum_a - a_mean * sum_b + sum_a * b_mean;
    sff = sum_aa - (a_mean + a_mean) * sum_a + sum_a * a_mean;
    smm = sum_bb - (b_mean + b_mean) * sum_b + sum_b * b_mean;
    """,
    name="cudipy_cc_precompute",
)


def convolve_separable(x, w, axes=None, **kwargs):
    """n-dimensional convolution via separable application of convolve1d

    Parameters
    ----------
    x : cupy.ndarray
        The input array.
    w : cupy.ndarray or sequence of cupy.ndarray
        If a single array is given, this same filter will be applied along
        all axes. A sequence of arrays can be provided in order to apply a
        separate filter along each axis. In this case the length of ``w`` must
        match the number of axes filtered.
    axes : tuple of int or None
        The axes of ``x`` to be filtered. The default (None) is to filter all
        axes of ``x``.

    Returns
    -------
    out : cupy.ndarray
        The filtered array.

    """
    if axes is None:
        axes = range(x.ndim)
    axes = tuple(axes)
    ndim = x.ndim
    if any(ax < -ndim or ax > ndim - 1 for ax in axes):
        raise ValueError("axis out of range")

    if isinstance(w, cp.ndarray):
        w = [w] * len(axes)
    elif len(w) != len(axes):
        raise ValueError("user should supply one filter per axis")

    for ax, w0 in zip(axes, w):
        if not isinstance(w0, cp.ndarray) or w0.ndim != 1:
            raise ValueError("w must be a 1d array (or sequence of 1d arrays)")
        x = ndi.convolve1d(x, w0, axis=ax, **kwargs)
    return x


def precompute_cc_factors(ad, bd, radius, mode="constant"):

    # factors = cp.zeros((5,) + ad.shape, dtype=ad.dtype)
    factors = [None] * 5
    sum_h = cp.ones((2 * radius + 1,), dtype=ad.dtype)
    h_tuple = (sum_h,) * ad.ndim
    kwargs = dict(mode=mode)
    sum_a = convolve_separable(ad, h_tuple, **kwargs)
    sum_b = convolve_separable(bd, h_tuple, **kwargs)
    sum_ab = convolve_separable(ad * bd, h_tuple, **kwargs)
    sum_aa = convolve_separable(ad * ad, h_tuple, **kwargs)
    sum_bb = convolve_separable(bd * bd, h_tuple, **kwargs)
    if mode != "constant":
        cnt = (2 * radius + 1) ** ad.ndim
    else:
        cnt = convolve_separable(
            cp.ones_like(ad), (sum_h,) * ad.ndim, **kwargs
        ).astype(cp.int32)

    if True:
        factors[0] = cp.empty_like(ad)
        factors[1] = cp.empty_like(ad)
        factors[2] = cp.empty_like(ad)
        factors[3] = cp.empty_like(ad)
        factors[4] = cp.empty_like(ad)
        _cc_precompute(
            ad,
            bd,
            sum_a,
            sum_b,
            sum_ab,
            sum_aa,
            sum_bb,
            cnt,
            factors[0],
            factors[1],
            factors[2],
            factors[3],
            factors[4],
        )
    else:
        a_mean = sum_a / cnt
        b_mean = sum_b / cnt
        factors[0] = ad - a_mean
        factors[1] = bd - b_mean
        factors[2] = sum_ab - b_mean * sum_a - a_mean * sum_b + sum_a * b_mean
        factors[3] = sum_aa - (a_mean + a_mean) * sum_a + sum_a * a_mean
        factors[4] = sum_bb - (b_mean + b_mean) * sum_b + sum_b * b_mean

    return factors


_cc_compute_forward = cp.ElementwiseKernel(
    in_params="W sfm, W sff, W smm, W Ji, W Ii",
    out_params="W out",
    operation="""
    W p = sff * smm;
    if (p == 0)
    {
        out = 0;
    } else {
        out =  -2.0 * sfm / p * (Ji - sfm / sff * Ii);
    }
    """,
    name="cudipy_cc_compute_forward",
)


_cc_compute_backward = cp.ElementwiseKernel(
    in_params="W sfm, W sff, W smm, W Ji, W Ii",
    out_params="W out",
    operation="""
    W p = sff * smm;
    if (p == 0)
    {
        out = 0;
    } else {
        out =  -2.0 * sfm / p * (Ii - sfm / smm * Ji);
    }
    """,
    name="cudipy_cc_compute_backward",
)


_cc_local_correlation = cp.ElementwiseKernel(
    in_params="W sfm, W sff, W smm, float64 thresh",
    out_params="W local_correlation",
    operation="""
    W p = sff * smm;
    if (p < thresh)
    {
        local_correlation = 0;
    } else {
        local_correlation =  (sfm * sfm) / (sff * smm);
        local_correlation > 1.0 ? 1.0 : local_correlation;
        local_correlation = -local_correlation;
    }
    """,
    name="cudipy_cc_compute_local_correlation",
)


def _compute_cc_step(
    grad_static, factors, radius, forward=True, zero_borders=True,
    coord_axis=-1
):
    out = cp.empty_like(grad_static)
    ndim = out.ndim - 1
    Ii = factors[0]
    Ji = factors[1]
    sfm = factors[2]
    sff = factors[3]
    smm = factors[4]

    result = cp.empty_like(sfm)
    _cc_local_correlation(sfm, sff, smm, 1e-5, result)
    energy = result.sum()

    if forward:
        cc_kernel = _cc_compute_forward
    else:
        cc_kernel = _cc_compute_backward
    # can reuse result for the output array
    cc_kernel(sfm, sff, smm, Ji, Ii, result)

    if coord_axis == -1:
        result = result[..., cp.newaxis]
    else:
        result = result[cp.newaxis, :]
    result = result * grad_static

    if zero_borders and radius > 0:
        if coord_axis not in [0, -1]:
            raise ValueError("coord_axis must be 0 or -1.")
        slices = [slice(None)] * (ndim + 1)
        if coord_axis == -1:
            axes = range(ndim)
        else:
            axes = range(1, ndim + 1)
        for ax in axes:
            slices[ax] = slice(0, radius)
            result[tuple(slices)] = 0
            slices[ax] = slice(-radius, None)
            result[tuple(slices)] = 0
            slices[ax] = slice(None)
    return result, energy


def compute_cc_forward_step(
    grad_static, factors, radius, zero_borders=True, coord_axis=-1
):
    return _compute_cc_step(
        grad_static=grad_static,
        factors=factors,
        radius=radius,
        forward=True,
        zero_borders=zero_borders,
        coord_axis=coord_axis,
    )


def compute_cc_backward_step(
    grad_static, factors, radius, zero_borders=True, coord_axis=-1
):
    return _compute_cc_step(
        grad_static=grad_static,
        factors=factors,
        radius=radius,
        forward=False,
        zero_borders=zero_borders,
        coord_axis=coord_axis,
    )
