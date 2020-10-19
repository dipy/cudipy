import cupy as cp

from .vector_fields import (          # noqa
    compose_vector_fields,
    gradient,
    invert_vector_field_fixed_point,
    reorient_vector_field,
    sparse_gradient,
    transform_affine,
    warp,
)

floating = cp.float32                 # noqa
del cp
