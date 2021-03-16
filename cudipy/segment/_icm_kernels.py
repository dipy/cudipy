import numpy

import cupy
from cudipy._vendored._cupy import _filters_core


@cupy.memoize()
def _get_icm_prob_class_kernel(w_shape, int_type):
    ndim = len(w_shape)
    # int_type = "size_t" if input.size > 1 << 31 else "int"

    found = """
    if ((int){value} == channel){{
        sum -= wval;
    }} else if ((int){value} >= 0) {{  // using mode='constant' with cval = -1 to skip neighbors outside the image extent
        sum += wval;
    }}
    """

    out_op = "y = cast<Y>(sum);"

    # origin=0 -> centered kernel
    offsets = _filters_core._origins_to_offsets([0] * ndim, w_shape)

    return _filters_core._generate_nd_kernel(
        name="icm_prob_class_given_neighb",
        pre="double sum = 0.;",
        found=found,
        post=out_op,
        mode="constant",
        w_shape=w_shape,
        int_type=int_type,
        offsets=offsets,
        cval=-1,  # cval
        has_weights=True,
        has_channel=True,
    )


def _get_icm_weights(ndim, beta, float_dtype):
    weights = numpy.zeros((3,) * ndim, dtype=float_dtype)
    if ndim == 2:
        weights[0, 1] = beta
        weights[0, -1] = beta
        weights[1, 0] = beta
        weights[-1, 0] = beta
    if ndim == 3:
        weights[0, 0, 1] = beta
        weights[0, 0, -1] = beta
        weights[0, 1, 0] = beta
        weights[0, -1, 0] = beta
        weights[1, 0, 0] = beta
        weights[-1, 0, 0] = beta
    else:
        raise ValueError("only 2D and 3D weights are implemented")
    return cupy.asarray(weights)
