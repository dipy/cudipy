import numpy as np

import cupy
from cupy import memoize
import cupyx.scipy.ndimage as ndi


# TODO: can generalize the following to nd using string templates
# Note: x, y, z be sparse coordinate arrays as returned by meshgrid with
#       sparse=True.
# reorient_kernel are equivalent to mul0 with last_col=0


reorient_kernel_3d = cupy.ElementwiseKernel(
    in_params="W x, W y, W z, raw W affine",
    out_params="W ox, W oy, W oz",
    operation="""
    ox = x*affine[0] + y*affine[1] + z*affine[2];
    oy = x*affine[3] + y*affine[4] + z*affine[5];
    oz = x*affine[6] + y*affine[7] + z*affine[8];
    """,
    name="cudipy_reorient_3d",
)


reorient_kernel_2d = cupy.ElementwiseKernel(
    in_params="W x, W y, raw W affine",
    out_params="W ox, W oy",
    operation="""
    ox = x*affine[0] + y*affine[1];
    oy = x*affine[2] + y*affine[3];
    """,
    name="cudipy_reorient_2d",
)


# aff_kernel are equivalent to mul0 with last_col=1
aff_kernel_3d = cupy.ElementwiseKernel(
    in_params="W x, W y, W z, raw W affine",
    out_params="W ox, W oy, W oz",
    operation="""
    ox = x*affine[0] + y*affine[1] + z*affine[2] + affine[3];
    oy = x*affine[4] + y*affine[5] + z*affine[6] + affine[7];
    oz = x*affine[8] + y*affine[9] + z*affine[10] + affine[11];
    """,
    name="cudipy_affine_3d",
)


aff_kernel_2d = cupy.ElementwiseKernel(
    in_params="W x, W y, raw W affine",
    out_params="W ox, W oy",
    operation="""
    ox = x*affine[0] + y*affine[1] + affine[2];
    oy = x*affine[3] + y*affine[4] + affine[5];
    """,
    name="cudipy_affine_2d",
)


# aff_kernel are equivalent to mul0 with last_col=1
composeAB_3d = cupy.ElementwiseKernel(
    in_params="W x, W y, W z, W x2, W y2, W z2, raw W affine1, raw W affine2",
    out_params="W ox, W oy, W oz",
    operation="""

    //affine1 applied without translation
    ox = x*affine1[0] + y*affine1[1] + z*affine1[2];
    oy = x*affine1[3] + y*affine1[4] + z*affine1[5];
    oz = x*affine1[6] + y*affine1[7] + z*affine1[8];

    //affine2 applied with translation and offset by tmpx
    ox += x2*affine2[0] + y2*affine2[1] + z2*affine2[2] + affine2[3];
    oy += x2*affine2[4] + y2*affine2[5] + z2*affine2[6] + affine2[7];
    oz += x2*affine2[8] + y2*affine2[9] + z2*affine2[10] + affine2[11];
    """,
    name="cudipy_composeAB_3d",
)


# aff_kernel are equivalent to mul0 with last_col=1
composeA_3d = cupy.ElementwiseKernel(
    in_params="W x2, W y2, W z2, raw W affine2",
    out_params="W ox, W oy, W oz",
    operation="""

    //affine2 applied with translation
    ox = x2*affine2[0] + y2*affine2[1] + z2*affine2[2] + affine2[3] + tmp1;
    oy = x2*affine2[4] + y2*affine2[5] + z2*affine2[6] + affine2[7] + tmp2;
    oz = x2*affine2[8] + y2*affine2[9] + z2*affine2[10] + affine2[11] + tmp3;
    """,
    name="cudipy_composeA_3d",
)


composeB_3d = cupy.ElementwiseKernel(
    in_params="W x, W y, W z, W x2, W y2, W z2, raw W affine1",
    out_params="W ox, W oy, W oz",
    operation="""

    //affine1 applied without translation
    ox = x*affine1[0] + y*affine1[1] + z*affine1[2];
    oy = x*affine1[3] + y*affine1[4] + z*affine1[5];
    oz = x*affine1[6] + y*affine1[7] + z*affine1[8];

    //offset by second set of coordinates
    ox += x2;
    oy += y2;
    oz += z2;
    """,
    name="cudipy_composeB_3d",
)


composeNone_3d = cupy.ElementwiseKernel(
    in_params="W x, W y, W z, W x2, W y2, W z2",
    out_params="W ox, W oy, W oz",
    operation="""

    //affine1 applied without translation
    ox = x + x2;
    oy = y + y2;
    oz = z + z2;
    """,
    name="cudipy_composeNone_3d",
)


# aff_kernel are equivalent to mul0 with last_col=1
composeAB_2d = cupy.ElementwiseKernel(
    in_params="W x, W y, W x2, W y2, raw W affine1, raw W affine2",
    out_params="W ox, W oy",
    operation="""

    //affine1 applied without translation
    ox = x*affine1[0] + y*affine1[1];
    oy = x*affine1[2] + y*affine1[3];

    //affine2 applied with translation and offset by tmpx
    ox += x2*affine2[0] + y2*affine2[1] + affine2[2];
    oy += x2*affine2[3] + y2*affine2[4] + affine2[5];
    """,
    name="cudipy_composeB_2d",
)


# aff_kernel are equivalent to mul0 with last_col=1
composeA_2d = cupy.ElementwiseKernel(
    in_params="W x2, W y2, raw W affine2",
    out_params="W ox, W oy",
    operation="""

    //affine2 applied with translation
    ox = x2*affine2[0] + y2*affine2[1] + affine2[2] + tmp1;
    oy = x2*affine2[3] + y2*affine2[4] + affine2[5] + tmp2;
    """,
    name="cudipy_composeA_2d",
)


composeB_2d = cupy.ElementwiseKernel(
    in_params="W x, W y, W x2, W y2, raw W affine1",
    out_params="W ox, W oy",
    operation="""

    //affine1 applied without translation
    ox = x*affine1[0] + y*affine1[1];
    oy = x*affine1[2] + y*affine1[3];

    //offset by second set of coordinates
    ox += x2;
    oy += y2;
    """,
    name="cudipy_composeB_2d",
)


composeNone_2d = cupy.ElementwiseKernel(
    in_params="W x, W y, W x2, W y2",
    out_params="W ox, W oy",
    operation="""

    //affine1 applied without translation
    ox = x + x2;
    oy = y + y2;
    """,
    name="cudipy_composeNone_2d",
)


def reorient_vector_field(field, affine, out=None, coord_axis=-1):
    return _apply_affine_to_field(
        field, affine, out, include_translations=False, coord_axis=coord_axis
    )


def _apply_affine_to_field(
    field, affine, out=None, include_translations=False, coord_axis=-1
):
    """Reorient a vector field.

    Parameters
    ----------
    field : cupy.ndarray
        The vector displacement field. Should have ndim + 1 dimensions with
        shape (ndim) on the first axis. Alternatively it can be a list of
        length `ndim` where each array corresponds to one coordinate
        vector. This type of list inputs allows for field to be a
        coordinate array as returned by meshgrid (optionally using the
        `sparse=True` option to conserve memory).
    affine : cupy.ndarray
        affine should be a transformation matrix with shape (ndim, ndim).
        It can also be a (ndim + 1, ndim + 1) affine matrix, but in this
        case,  only the upper left (ndim, ndim) portion of the matrix will
        be applied.
    out : cupy.ndarray or None
        The output array (same shape as `field`). Note that in-place
        reorientation is not supported (i.e. `out` cannot be the same array
        as `field`).

    Returns
    -------
    reoriented : cupy.ndarray
        The reoriented displacement field.

    """
    # TODO: remove support for dimensions as last axis instead of first?
    #       that would simplify the function. first axis is more efficient for
    #       C-contiguous arrays
    if coord_axis not in [0, -1]:
        raise ValueError("coord_axis must be 0 or -1.")
    if isinstance(field, cupy.ndarray):
        # field is a single, dense ndarray
        ndim = field.ndim - 1
        if field.shape[coord_axis] != ndim:
            print(
                f"field.shape={field.shape}, ndim={ndim}, coord_axis={coord_axis}"
            )
            raise ValueError("shape mismatch")

        if field.ndim != ndim + 1:
            raise ValueError("invalid field")
        if not field.dtype.kind == "f":
            raise ValueError("field must having floating point dtype")
        if coord_axis == 0:
            field = tuple([f for f in field])
        else:
            field = tuple(
                [cupy.ascontiguousarray(field[..., n]) for n in range(ndim)]
            )
    else:
        ndim = len(field)
    field_dtype = field[0].dtype

    if include_translations:
        affine_shape = (ndim, ndim + 1)
    else:
        affine_shape = (ndim, ndim)
    affine = cupy.asarray(affine, dtype=field_dtype, order="C")
    if affine.shape == (ndim + 1, ndim + 1):
        affine = affine[: affine_shape[0], : affine_shape[1]]
        if not affine.flags.c_contiguous:
            affine = cupy.ascontiguousarray(affine)
    if affine.shape != affine_shape:
        raise ValueError(
            "expected anaffine array with shape {}".format(affine_shape)
        )

    out_shape = (ndim,) + tuple([field[n].shape[n] for n in range(ndim)])
    if out is None:
        out = cupy.empty(out_shape, dtype=field_dtype, order="C")
    else:
        if out.shape != field.shape or out.dtype != field_dtype:
            raise ValueError("out and field must have matching shape and dtype")
        if not out.flags.c_contiguous:
            raise ValueError("out must be C contiguous")

    # Note: affine must be contiguous.
    #       field and out do not have to be contiguous, but performance
    #       will be better if they are.
    if ndim == 3:
        if include_translations:
            kernel = aff_kernel_3d
        else:
            kernel = reorient_kernel_3d
        # args = field + (affine, out[0], out[1], out[2])
        kernel(*field, affine, out[0], out[1], out[2])
    elif ndim == 2:
        if include_translations:
            kernel = aff_kernel_2d
        else:
            kernel = reorient_kernel_2d
        kernel(*field, affine, out[0], out[1])

    if coord_axis == -1:
        out = cupy.moveaxis(out, 0, -1)
    return out


# y stores the coordinate along a given axis
# for locations where y > 0 and y < size:
#    res = d + time_scale * z
# count will be set non-zero at any location outside the bounds
# _comp_apply_masked_time_scaling_nd has to be called independently for each
# dimension
_comp_apply_masked_time_scaling_nd = cupy.ElementwiseKernel(
    in_params="W d, W y, W z, W time_scale, int32 size",
    out_params="W res, int32 count",
    operation="""
    if ((y >= 0) && (y < size))
    {
        res = d + time_scale * z;
    } else {
        res = 0;
        atomicAdd((int*)&count, 1);
    }
    """,
    name="cudipy_masked_time_scaling",
)


_comp_apply_masked_time_scaling_2d = cupy.ElementwiseKernel(
    in_params="W d1, W d2, W y1, W y2, W z1, W z2, W time_scale, raw int32 size",
    out_params="W res1, W res2",
    operation="""
    if ((y1 >= 0) && (y1 < size[0]) && (y2 >= 0) && (y2 < size[1]))
    {
        res1 = d1 + time_scale * z1;
        res2 = d2 + time_scale * z2;
    } else {
        res1 = 0;
        res2 = 0;
    }
    """,
    name="cudipy_masked_time_scaling_2d",
)


_comp_apply_masked_time_scaling_3d = cupy.ElementwiseKernel(
    in_params="W d1, W d2, W d3, W y1, W y2, W y3, W z1, W z2, W z3, W time_scale, raw int32 size",
    out_params="W res1, W res2, W res3",
    operation="""
    if ((y1 >= 0) && (y1 < size[0]) && (y2 >= 0) && (y2 < size[1]) && (y3 >= 0) && (y3 < size[2]))
    {
        res1 = d1 + time_scale * z1;
        res2 = d2 + time_scale * z2;
        res3 = d3 + time_scale * z3;
    } else {
        res1 = 0;
        res2 = 0;
        res3 = 0;
    }
    """,
    name="cudipy_masked_time_scaling_3d",
)


def compose_vector_fields(
    d1,
    d2,
    premult_index,
    premult_disp,
    time_scaling,
    comp=None,
    order=1,
    *,
    coord_axis=-1,
    omit_stats=False,
    xcoords=None,
    Y=None,
    Z=None,
):
    if comp is None:
        comp = cupy.empty_like(d1, order="C")

    # need vector elements on first axis, not last
    if coord_axis != 0:
        d1 = cupy.ascontiguousarray(cupy.moveaxis(d1, -1, 0))
        d2 = cupy.ascontiguousarray(cupy.moveaxis(d2, -1, 0))
    else:
        if not d1.flags.c_contiguous:
            d1 = cupy.ascontiguousarray(d1)
        if not d2.flags.c_contiguous:
            d2 = cupy.ascontiguousarray(d2)
    ndim = d1.shape[0]
    B = premult_disp
    A = premult_index
    t = time_scaling

    if xcoords is None:
        xcoords = cupy.meshgrid(
            *[cupy.arange(s, dtype=d1.real.dtype) for s in d1.shape[1:]],
            indexing="ij",
            sparse=True,
        )

    # TODO: reduce number of temporary arrays
    if ndim in [2, 3]:
        if Y is None:
            Y = cupy.empty_like(d1)
        if A is None:
            if B is None:
                if ndim == 3:
                    composeNone_3d(
                        d1[0],
                        d1[1],
                        d1[2],
                        xcoords[0],
                        xcoords[1],
                        xcoords[2],
                        Y[0],
                        Y[1],
                        Y[2],
                    )
                else:
                    composeNone_2d(
                        d1[0], d1[1], xcoords[0], xcoords[1], Y[0], Y[1]
                    )
            else:
                B = cupy.asarray(B[:ndim, :ndim], dtype=d1.dtype, order="C")
                if ndim == 3:
                    composeB_3d(
                        d1[0],
                        d1[1],
                        d1[2],
                        xcoords[0],
                        xcoords[1],
                        xcoords[2],
                        B,
                        Y[0],
                        Y[1],
                        Y[2],
                    )
                else:
                    composeB_2d(
                        d1[0], d1[1], xcoords[0], xcoords[1], B, Y[0], Y[1]
                    )
        elif B is None:
            A = cupy.asarray(A[:ndim, :], dtype=d1.dtype, order="C")
            if ndim == 3:
                composeA_3d(
                    xcoords[0], xcoords[1], xcoords[2], A, Y[0], Y[1], Y[2]
                )
            else:
                composeA_2d(xcoords[0], xcoords[1], A, Y[0], Y[1])
        else:
            A = cupy.asarray(A[:ndim, :], dtype=d1.dtype, order="C")
            B = cupy.asarray(B[:ndim, :ndim], dtype=d1.dtype, order="C")
            if ndim == 3:
                composeAB_3d(
                    d1[0],
                    d1[1],
                    d1[2],
                    xcoords[0],
                    xcoords[1],
                    xcoords[2],
                    B,
                    A,
                    Y[0],
                    Y[1],
                    Y[2],
                )
            else:
                composeAB_2d(
                    d1[0], d1[1], xcoords[0], xcoords[1], B, A, Y[0], Y[1]
                )
    else:
        if B is None:
            d1tmp = d1.copy()  # have to copy to avoid modification of d1
        else:
            d1tmp = _apply_affine_to_field(
                d1, B[:ndim, :ndim], include_translations=False, coord_axis=0
            )

        if A is None:
            Y = d1tmp
            for n in range(ndim):
                Y[n] += xcoords[n]
        else:
            # Y = mul0(A, xcoords, sh, cupy, lastcol=1)
            Y = _apply_affine_to_field(
                xcoords, A[:ndim, :], include_translations=True, coord_axis=0
            )
            Y += d1tmp

    if Z is None:
        Z = cupy.empty_like(Y)
    for n in range(ndim):
        Z[n, ...] = ndi.map_coordinates(d2[n], Y, order=1, mode="constant")

    if coord_axis == 0:
        res = comp
    else:
        res = cupy.empty_like(Z)

    if omit_stats and ndim in [2, 3]:
        _shape = cupy.asarray(
            [d1.shape[1 + n] - 1 for n in range(ndim)], dtype=cupy.int32
        )
        if ndim == 3:
            _comp_apply_masked_time_scaling_3d(
                d1[0],
                d1[1],
                d1[2],
                Y[0],
                Y[1],
                Y[2],
                Z[0],
                Z[1],
                Z[2],
                t,
                _shape,
                res[0],
                res[1],
                res[2],
            )
        else:
            _comp_apply_masked_time_scaling_2d(
                d1[0], d1[1], Y[0], Y[1], Z[0], Z[1], t, _shape, res[0], res[1]
            )
    else:

        # TODO: declare count as boolean?
        count = cupy.zeros(Z.shape[1:], dtype=np.int32)

        # We now compute:
        #    res = d1 + t * Z
        #    except that res = 0 where either coordinate in
        #    interpolating Y was outside the displacement extent
        for n in range(ndim):
            _comp_apply_masked_time_scaling_nd(
                d1[n], Y[n], Z[n], t, d1.shape[1 + n] - 1, res[n], count
            )

        # nnz corresponds to the number of points in comp inside the domain
        count = count > 0  # remove after init count as boolean
        if not omit_stats:
            nnz = res.size // ndim - cupy.count_nonzero(count)
        res *= ~count[np.newaxis, ...]

    if omit_stats:
        stats = None
    else:
        # compute the stats
        stats = cupy.empty((3,), dtype=float)
        nn = res[0] * res[0]
        for n in range(1, ndim):
            nn += res[n] * res[n]
        # TODO: do we want stats to be a GPU array or CPU array?
        stats[0] = cupy.sqrt(nn.max())
        mean_norm = nn.sum() / nnz
        stats[1] = cupy.sqrt(mean_norm)
        nn *= nn
        stats[2] = cupy.sqrt(nn.sum() / nnz - mean_norm * mean_norm)

    if coord_axis != 0:
        res = cupy.moveaxis(res, 0, -1)
        comp[...] = res

    return comp, stats


def simplify_warp_function(
    d,
    affine_idx_in,
    affine_idx_out,
    affine_disp,
    out_shape,
    *,
    mode="constant",
    coord_axis=-1,
):
    """
    Simplifies a nonlinear warping function combined with an affine transform

    Modifies the given deformation field by incorporating into it
    an affine transformation and voxel-to-space transforms associated with
    the discretization of its domain and codomain.

    The resulting transformation may be regarded as operating on the
    image spaces given by the domain and codomain discretization.
    More precisely, the resulting transform is of the form:

    (1) T[i] = W * d[U * i] + V * i

    Where U = affine_idx_in, V = affine_idx_out, W = affine_disp.

    Parameters
    ----------
    d : array, shape (S', R', C', 3)
        the non-linear part of the transformation (displacement field)
    affine_idx_in : array, shape (4, 4)
        the matrix U in eq. (1) above
    affine_idx_out : array, shape (4, 4)
        the matrix V in eq. (1) above
    affine_disp : array, shape (4, 4)
        the matrix W in eq. (1) above
    out_shape : array, shape (3,)
        the number of slices, rows and columns of the sampling grid

    Returns
    -------
    out : array, shape = out_shape
        the deformation field `out` associated with `T` in eq. (1) such that:
        T[i] = i + out[i]

    Notes
    -----
    Both the direct and inverse transforms of a DiffeomorphicMap can be written
    in this form:

    Direct:  Let D be the voxel-to-space transform of the domain's
             discretization, P be the pre-align matrix, Rinv the space-to-voxel
             transform of the reference grid (the grid the displacement field
             is defined on) and Cinv be the space-to-voxel transform of the
             codomain's discretization. Then, for each i in the domain's grid,
             the direct transform is given by

             (2) T[i] = Cinv * d[Rinv * P * D * i] + Cinv * P * D * i

             and we identify U = Rinv * P * D, V = Cinv * P * D, W = Cinv

    Inverse: Let C be the voxel-to-space transform of the codomain's
             discretization, Pinv be the inverse of the pre-align matrix, Rinv
             the space-to-voxel transform of the reference grid (the grid the
             displacement field is defined on) and Dinv be the space-to-voxel
             transform of the domain's discretization. Then, for each j in the
             codomain's grid, the inverse transform is given by

             (3) Tinv[j] = Dinv * Pinv * d[Rinv * C * j] + Dinv * Pinv * C * j

             and we identify U = Rinv * C, V = Dinv * Pinv * C, W = Dinv * Pinv

    """
    if coord_axis not in [0, -1]:
        raise ValueError("coord_axis must be 0 or -1")
    ndim = d.shape[coord_axis]
    U = affine_idx_in
    V = affine_idx_out
    W = affine_disp
    # TODO: reduce number of temporary arrays
    coord_dtype = cupy.promote_types(d.dtype, np.float32)
    if U is None:
        xcoords = cupy.meshgrid(
            *[cupy.arange(s, dtype=coord_dtype) for s in d.shape[:-1]],
            indexing="ij",
            sparse=True,
        )
        if coord_axis == 0:
            Z = d.copy()
        else:
            Z = cupy.ascontiguousarray(cupy.moveaxis(d, -1, 0))
    else:
        xcoords = cupy.meshgrid(
            *[cupy.arange(s, dtype=coord_dtype) for s in d.shape[:-1]],
            indexing="ij",
            sparse=True,
        )
        # Y = mul0(A, xcoords, sh, cupy, lastcol=1)
        Y = _apply_affine_to_field(
            xcoords,
            U[:ndim, :],
            out=None,
            include_translations=True,
            coord_axis=0,
        )

        # for CuPy with non-legacy linear interpolation, don't need to extend d
        Z = cupy.empty_like(Y)
        if coord_axis == 0:
            for n in range(ndim):
                Z[n, ...] = ndi.map_coordinates(d[n], Y, order=1, mode=mode)
        else:
            for n in range(ndim):
                Z[n, ...] = ndi.map_coordinates(
                    d[..., n], Y, order=1, mode=mode
                )

    if W is not None:
        # Z = mul0(C, Z, sh, cupy, out=Z, lastcol=0)
        Z = _apply_affine_to_field(
            Z,
            W[:ndim, :ndim],
            out=None,
            include_translations=False,
            coord_axis=0,
        )

    if V is not None:
        Z += _apply_affine_to_field(
            xcoords,
            V[:ndim, :],
            out=None,
            include_translations=True,
            coord_axis=0,
        )
        for n in range(ndim):
            Z[n, ...] -= xcoords[
                n
            ]  # TODO: just subtract one from last column of V instead?
    if coord_axis == -1:
        Z = cupy.moveaxis(Z, 0, -1)
    if not Z.flags.c_contiguous:
        Z = cupy.ascontiguousarray(Z)
    return Z


"""
Elementwise kernel to do the following operation efficiently:

where norms > maxlen:
    p = p - epsilon * maxlen / norms * q
else:
    p = p - epsilon * q

shape of p, q will be (nx, ny, nz, 3) in 3D
shape of norms input should be (nx, ny, nz, 1) in 3D
ElementWise kernel will take of broadcasting the norms
epsilon and maxlen are scalars
"""
_norm_thresh_kernel = cupy.ElementwiseKernel(
    in_params="W q, W norms, W epsilon, W maxlen",
    out_params="W p",
    operation="""
    double step_factor;
    if (norms > maxlen)
    {
        step_factor = epsilon * maxlen / norms;
    } else {
        step_factor = epsilon;
    }
    p = p - step_factor * q;
    """,
    name="cudipy_norm_thresh",
)


def invert_vector_field_fixed_point(
    d,
    d_world2grid,
    spacing,
    max_iter,
    tol,
    start=None,
    *,
    coord_axis=-1,
    print_stats=False,
):
    """Computes the inverse of a 3D displacement fields

    Computes the inverse of the given 3-D displacement field d using the
    fixed-point algorithm [1].

    [1] Chen, M., Lu, W., Chen, Q., Ruchala, K. J., & Olivera, G. H. (2008).
        A simple fixed-point approach to invert a deformation field.
        Medical Physics, 35(1), 81. doi:10.1118/1.2816107

    Parameters
    ----------
    d : array, shape (S, R, C, 3)
        the 3-D displacement field to be inverted
    d_world2grid : array, shape (4, 4)
        the space-to-grid transformation associated to the displacement field
        d (transforming physical space coordinates to voxel coordinates of the
        displacement field grid)
    spacing :array, shape (3,)
        the spacing between voxels (voxel size along each axis)
    max_iter : int
        maximum number of iterations to be performed
    tol : float
        maximum tolerated inversion error
    start : array, shape (S, R, C)
        an approximation to the inverse displacement field (if no approximation
        is available, None can be provided and the start displacement field
        will be zero)

    Returns
    -------
    p : array, shape (S, R, C, 3)
        the inverse displacement field

    Notes
    -----
    We assume that the displacement field is an endomorphism so that the shape
    and voxel-to-space transformation of the inverse field's discretization is
    the same as those of the input displacement field. The 'inversion error' at
    iteration t is defined as the mean norm of the displacement vectors of the
    input displacement field composed with the inverse at iteration t.
    """

    ndim = d.shape[coord_axis]
    if coord_axis != 0:
        d = cupy.moveaxis(d, coord_axis, 0)

    if start is not None:
        if coord_axis != 0:
            start = cupy.moveaxis(start, coord_axis, 0)
            if start.shape != d.shape:
                raise ValueError("start must have the same shape as d")
            p = cupy.ascontiguousarray(start)
        else:
            p = start.copy()
    else:
        p = cupy.zeros_like(d)
    q = cupy.empty_like(d)

    if spacing.dtype != q.dtype:
        spacing = spacing.astype(q.dtype)

    if d_world2grid is not None:
        d_world2grid = cupy.asarray(d_world2grid)
        if not d_world2grid.flags.c_contiguous:
            d_world2grid = cupy.ascontiguousarray(d_world2grid)

    # for efficiency, precompute xcoords, Y and Z here instead of repeatedly
    # doing so inside of compose_vector_fields
    xcoords = cupy.meshgrid(
        *[cupy.arange(s, dtype=d.real.dtype) for s in d.shape[1:]],
        indexing="ij",
        sparse=True,
    )
    Y = cupy.empty_like(d)
    Z = cupy.empty_like(d)

    iter_count = 0
    difmag = 1
    error = 1 + tol
    while (0.1 < difmag) and (iter_count < max_iter) and (tol < error):
        if iter_count == 0:
            epsilon = 0.75
        else:
            epsilon = 0.5
        q, _stats = compose_vector_fields(
            p,
            d,
            None,
            d_world2grid,
            1.0,
            comp=q,
            coord_axis=0,
            omit_stats=True,
            xcoords=xcoords,
            Y=Y,
            Z=Z,
        )
        difmag = 0
        error = 0

        # could make special case 2d/3d elementwise kernel for computing norms
        norms = q[0] / spacing[0]
        norms *= norms
        for n in range(1, ndim):
            tmp = q[n] / spacing[n]
            norms += tmp * tmp
        cupy.sqrt(norms, out=norms)

        error = float(norms.sum())
        difmag = float(norms.max())
        maxlen = difmag * epsilon
        _norm_thresh_kernel(q, norms[np.newaxis, ...], epsilon, maxlen, p)

        error /= norms.size
        iter_count += 1

    if print_stats:
        stats = np.empty((2,), dtype=float)
        stats[0] = error
        stats[1] = iter_count
        print(f"stats={stats}, diffmag={difmag}")
    if coord_axis != 0:
        p = cupy.moveaxis(p, 0, coord_axis)
    return p  # , q, norms, tmp1, tmp2, epsilon, maxlen


def warp(
    volume,
    d1,
    affine_idx_in=None,
    affine_idx_out=None,
    affine_disp=None,
    out_shape=None,
    *,
    order=1,
    mode="constant",
    coord_axis=-1,
):
    """
    Deforms the input volume under the given transformation. The warped volume
    is computed using is given by:

    (1) warped[i] = volume[ C * d1[A*i] + B*i ]

    where:
    A = affine_idx_in
    B = affine_idx_out
    C = affine_disp
    """
    A = affine_idx_in
    B = affine_idx_out
    C = affine_disp
    if out_shape is None:
        out_shape = volume.shape
    if A is not None:
        A = cupy.asarray(A)
    if B is not None:
        B = cupy.asarray(B)
    if C is not None:
        C = cupy.asarray(C)

    # TODO: reduce number of temporary arrays
    coord_dtype = cupy.promote_types(volume.dtype, np.float32)
    ndim = volume.ndim
    if d1.shape[coord_axis] != ndim:
        raise ValueError(
            "expected a displacement field with shape "
            "{} along axis {}".format(ndim, coord_axis)
        )
    if A is None:
        xcoords = cupy.meshgrid(
            *[cupy.arange(s, dtype=coord_dtype) for s in out_shape],
            indexing="ij",
            sparse=True,
        )
        Z = cupy.ascontiguousarray(cupy.moveaxis(d1, -1, 0))
    else:
        xcoords = cupy.meshgrid(
            *[cupy.arange(s, dtype=coord_dtype) for s in out_shape],
            indexing="ij",
            sparse=True,
        )
        # Y = mul0(A, xcoords, sh, cupy, lastcol=1)
        Y = _apply_affine_to_field(
            xcoords,
            A[:ndim, :],
            out=None,
            include_translations=True,
            coord_axis=0,
        )

        # for CuPy with non-legacy linear interpolation, don't need to extend d1
        Z = cupy.empty_like(Y)
        if coord_axis == -1:
            for n in range(ndim):
                Z[n, ...] = ndi.map_coordinates(
                    d1[..., n], Y, order=1, mode=mode
                )
        else:
            for n in range(ndim):
                Z[n, ...] = ndi.map_coordinates(d1[n], Y, order=1, mode=mode)

    if C is not None:
        # Z = mul0(C, Z, sh, cupy, out=Z, lastcol=0)
        Z = _apply_affine_to_field(
            Z,
            C[:ndim, :ndim],
            out=None,
            include_translations=False,
            coord_axis=0,
        )
    if B is not None:
        # Z += mul0(B, xcoords, sh, cupy, lastcol=1)
        Z += _apply_affine_to_field(
            xcoords,
            B[:ndim, :],
            out=None,
            include_translations=True,
            coord_axis=0,
        )
    else:
        if A is None:
            for n in range(ndim):
                Z[n, ...] += xcoords[n]
        else:
            Z += Y
    return ndi.map_coordinates(volume, Z, order=order, mode=mode)


def down2_ax(arr, axis):
    # TODO: doesn't handle odd sizes quite the same
    import itertools

    ndim = arr.ndim
    if axis < -arr.ndim or axis > arr.ndim - 1:
        raise ValueError("axis out of range")
    axis = axis % arr.ndim

    odd_size = arr.shape[axis] % 2

    sln = slice(None)
    sl0 = slice(None, None, 2)
    if odd_size:
        sl_out = [sln] * ndim
        sl_out[axis] = slice(None, -1)
        sl_out = tuple(sl_out)
    sl1 = slice(1, None, 2)
    offsets = [(sl0, sl1) if ax == axis else (sln,) for ax in range(ndim)]
    for n, offset_slices in enumerate(itertools.product(*offsets)):
        if n == 0:
            out = arr[offset_slices].copy()
        else:
            if odd_size:
                out[sl_out] += arr[offset_slices]
            else:
                out += arr[offset_slices]
    if odd_size:
        out[sl_out] /= 2
    else:
        out /= 2
    return out


def down2(arr, axes=None):
    # TODO: doesn't handle odd sizes quite the same
    ndim = arr.ndim
    if axes is None:
        axes = tuple(range(ndim))
    even_axes = []
    odd_axes = []
    for ax in axes:
        if arr.shape[ax] % 2:
            odd_axes.append(ax)
        else:
            even_axes.append(ax)
    if even_axes:
        out = down2_even(arr, axes=even_axes)
    else:
        out = arr
    for ax in odd_axes:
        out = down2_ax(out, axis=ax)
    return out


def down2_even(arr, axes=None, out=None):
    # Downsampling arr by two.
    # arr must have even size along all axes in `axes`.
    import itertools

    ndim = arr.ndim
    if axes is None:
        axes = tuple(range(ndim))
    out_shape = list(arr.shape)
    out_shape = tuple(
        [
            arr.shape[ax] // 2 if ax in axes else arr.shape[ax]
            for ax in range(ndim)
        ]
    )
    if out is None:
        out = cupy.empty(out_shape, dtype=arr.dtype)
    elif out.shape != out_shape:
        raise ValueError(
            "out does not have expected shape ({})".format(out_shape)
        )
    sl_even = slice(None, None, 2)
    sl_odd = slice(1, None, 2)
    sl_all = slice(None)
    offsets = [
        (sl_even, sl_odd) if ax in axes else (sl_all,) for ax in range(ndim)
    ]
    for offset_slices in itertools.product(*offsets):
        out += arr[offset_slices]
    out /= 2 ** len(axes)
    return out


def transform_affine(volume, ref_shape, affine, order=1):

    ndim = volume.ndim
    affine = cupy.asarray(affine)
    if True:
        out = ndi.affine_transform(
            volume,
            matrix=affine,
            order=order,
            mode="constant",
            output_shape=tuple(ref_shape),
        )
    else:
        # use map_coordinates instead of affine_transform
        xcoords = cupy.meshgrid(
            *[cupy.arange(s, dtype=volume.dtype) for s in ref_shape],
            indexing="ij",
            sparse=True,
        )
        coords = _apply_affine_to_field(
            xcoords,
            affine[:ndim, :],
            include_translations=True,
            coord_axis=0,
        )
        out = ndi.map_coordinates(volume, coords, order=1)

    return out


def resample_displacement_field(
    field,
    factors,
    out_shape,
    *,
    order=1,
    mode="constant",
    coord_axis=-1,
    output=None,
):
    """Resamples a 3D vector field to a custom target shape

    Resamples the given 3D displacement field on a grid of the requested shape,
    using the given scale factors. More precisely, the resulting displacement
    field at each grid cell i is given by

    D[i] = field[Diag(factors) * i]

    Parameters
    ----------
    factors : cp.ndarray
        the scaling factors mapping (integer) grid coordinates in the resampled
        grid to (floating point) grid coordinates in the original grid
    out_shape : tuple of int
        the desired shape of the resulting grid

    Returns
    -------
    expanded : array, shape = out_shape + (ndim, )
        the resampled displacement field
    """
    out_shape = tuple(out_shape)
    factors = cupy.asarray(
        factors, dtype=cupy.promote_types(field, cupy.float32)
    )
    ndim = field.shape[coord_axis]
    if coord_axis == -1:
        output = cupy.empty(out_shape + (ndim,), dtype=field.dtype)
        for n in range(ndim):
            output[..., n] = ndi.affine_transform(
                field[..., n],
                matrix=factors,
                output_shape=out_shape,
                order=order,
                mode=mode,
            )
    elif coord_axis == 0:
        output = cupy.empty((ndim,) + out_shape, dtype=field.dtype)
        for n in range(ndim):
            output[n] = ndi.affine_transform(
                field[n],
                matrix=factors,
                output_shape=out_shape,
                order=order,
                mode=mode,
            )
    else:
        raise ValueError("coord_axis must be 0 or -1.")
    return output


def _get_coord_dual_affine_grad(ndim, nprepad):
    """Compute target coordinate as in dipy.align.vector_fields.gradient_3d

    The homogeneous matrix has shape (ndim, ndim + 1). It corresponds to
    affine matrix where the last row of the affine is assumed to be:
    ``[0] * ndim + [1]``.

    Notes
    -----
    Assumes the following variables have been initialized on the device::

        mat(array): array containing the (ndim, ndim + 1) out_grid2world matrix.
        mat2(array): array containing the (ndim, ndim + 1) img_world2grid matrix.
        dx(array): array containing the pixel spacings
        in_coords(array): coordinates of the input

    For example, in 2D:

        tmp_0 = mat[0] * in_coords[0] + mat[1] * in_coords[1] + aff[2];
        tmp_1 = mat[3] * in_coords[0] + mat[4] * in_coords[1] + aff[5];

        tmp_0 += dx[0];
        tmp_1 += dx[1];

        c_0 = mat[0] * tmp_0 + mat[1] * tmp_1 + aff[2];
        c_1 = mat[3] * tmp_0 + mat[4] * tmp_1 + aff[5];

    """
    if nprepad != 0:
        raise NotImplementedError("nprepad not implemented")
    ops = []
    ncol = ndim + 1
    for j in range(ndim):
        ops.append(
            """
            W tmp_c_{j} = (W)0.0;
            """.format(
                j=j
            )
        )
        for k in range(ndim):
            m_index = ncol * j + k
            ops.append(
                """
            tmp_c_{j} += mat[{m_index}] * (W)in_coord[{k}];
                """.format(
                    j=j, k=k, m_index=m_index
                )
            )
        ops.append(
            """
            tmp_c_{j} += mat[{m_index}] + dx[{j}];
            """.format(
                j=j, m_index=ncol * j + ndim
            )
        )

    for j in range(ndim):
        ops.append(
            """
            W c_{j} = (W)0.0;
            """.format(
                j=j
            )
        )
        for k in range(ndim):
            m_index = ncol * j + k
            ops.append(
                """
            c_{j} += mat2[{m_index}] * (W)tmp_c_{k};
                """.format(
                    j=j, k=k, m_index=m_index
                )
            )
        ops.append(
            """
            c_{j} += mat2[{m_index}];
            """.format(
                j=j, m_index=ncol * j + ndim
            )
        )
        ops.append(
            """
            if ((c_{j} < 0) || (c_{j} >= (xsize_{j} - 1)))
            {{
                inside[i] = (I)0;
            }}""".format(
                j=j
            )
        )
    return ops


@memoize(for_each_device=True)
def _get_grad_kernel(
    ndim, large_int, yshape, mode, cval=0.0, order=1, integer_output=False
):
    from cudipy._vendored._cupy._interp_kernels import _generate_interp_custom

    in_params = "raw X x, raw W mat, raw W mat2, raw W dx"
    out_params = "Y y, raw I inside"
    operation, name = _generate_interp_custom(
        # in_params=in_params,
        coord_func=_get_coord_dual_affine_grad,
        ndim=ndim,
        large_int=large_int,
        yshape=yshape,
        mode=mode,
        cval=cval,
        order=order,
        name="cudipy_affine_grad",
        integer_output=integer_output,
    )
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)


def _get_coord_dual_affine_sparse_grad(ndim, nprepad):
    """Compute target coordinate as in dipy.align.vector_fields.gradient_3d

    The homogeneous matrix has shape (ndim, ndim + 1). It corresponds to
    affine matrix where the last row of the affine is assumed to be:
    ``[0] * ndim + [1]``.

    Notes
    -----
    Assumes the following variables have been initialized on the device::

        coords(array): array containing the coordinates of the output array at which to evaluate
        mat2(array): array containing the (ndim, ndim + 1) img_world2grid matrix.
        dx(array): array containing the pixel spacings
        in_coords(array): coordinates of the input

    For example, in 2D:

        tmp_0 = mat[0] * in_coords[0] + mat[1] * in_coords[1] + aff[2];
        tmp_1 = mat[3] * in_coords[0] + mat[4] * in_coords[1] + aff[5];

        tmp_0 += dx[0];
        tmp_1 += dx[1];

        c_0 = mat[0] * tmp_0 + mat[1] * tmp_1 + aff[2];
        c_1 = mat[3] * tmp_0 + mat[4] * tmp_1 + aff[5];

    """
    if nprepad != 0:
        raise NotImplementedError("nprepad not implemented")
    ops = []
    ncol = ndim + 1
    ops.append("ptrdiff_t ncoords = _ind.size();")
    for j in range(ndim):
        ops.append(
            """
    W tmp_c_{j} = coords[i + {j} * ncoords] + dx[{j}];""".format(
                j=j
            )
        )
    for j in range(ndim):
        ops.append(
            """
            W c_{j} = (W)0.0;""".format(
                j=j
            )
        )
        for k in range(ndim):
            m_index = ncol * j + k
            ops.append(
                """
            c_{j} += mat[{m_index}] * (W)tmp_c_{k};""".format(
                    j=j, k=k, m_index=m_index
                )
            )
        ops.append(
            """
            c_{j} += mat[{m_index}];
            """.format(
                j=j, m_index=ncol * j + ndim
            )
        )
        ops.append(
            """
            if ((c_{j} < 0) || (c_{j} >= (xsize_{j} - 1)))
            {{
                inside[i] = (I)0;
            }}""".format(
                j=j
            )
        )
    return ops


@memoize(for_each_device=True)
def _get_sparse_grad_kernel(
    ndim, large_int, yshape, mode, cval=0.0, order=1, integer_output=False
):
    from cudipy._vendored._cupy._interp_kernels import _generate_interp_custom

    in_params = "raw X x, raw W mat, raw W coords, raw W dx"
    out_params = "Y y, raw I inside"
    operation, name = _generate_interp_custom(
        # in_params=in_params,
        coord_func=_get_coord_dual_affine_sparse_grad,
        ndim=ndim,
        large_int=large_int,
        yshape=yshape,
        mode=mode,
        cval=cval,
        order=order,
        name="cudipy_sparse_affine_grad",
        integer_output=integer_output,
    )
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)


def gradient(
    img,
    img_world2grid,
    img_spacing,
    out_shape,
    out_grid2world,
    *,
    mode="constant",
    order=1,
    coord_axis=-1,
):
    # TODO: use inside?
    ndim = img.ndim
    large_int = False
    grad_kernel = _get_grad_kernel(
        ndim,
        large_int,
        img.shape,
        mode,
        cval=0.0,
        order=order,
        integer_output=False,
    )
    dx = cupy.zeros(ndim)
    tmp = cupy.empty_like(img)
    if coord_axis == 0:
        out_shape = (ndim,) + out_shape
    elif coord_axis == -1:
        out_shape = out_shape + (ndim,)
    else:
        raise ValueError("invalid coordinate axis")
    out = cupy.empty(out_shape, dtype=img.dtype)
    inside = cupy.ones(img.shape, dtype=np.int32)
    for ax in range(img.ndim):
        dx[ax] = -0.5 * img_spacing[ax]

        grad_kernel(img, out_grid2world, img_world2grid, dx, tmp, inside)
        dx[ax] = 0.5 * img_spacing[ax]
        if coord_axis == 0:
            grad_kernel(
                img, out_grid2world, img_world2grid, dx, out[ax], inside
            )
            out[ax] -= tmp
            out[ax] /= img_spacing[ax]
        else:
            grad_kernel(
                img, out_grid2world, img_world2grid, dx, out[..., ax], inside
            )
            out[..., ax] -= tmp
            out[..., ax] /= img_spacing[ax]
        dx[ax] = 0
    return out, inside


def sparse_gradient(
    img,
    img_world2grid,
    img_spacing,
    sample_points,
    out=None,
    inside=None,
    *,
    mode="constant",
    order=1,
    coord_axis=-1,
):
    # TODO: use inside?
    ndim = img.ndim
    large_int = False
    grad_kernel = _get_sparse_grad_kernel(
        ndim,
        large_int,
        img.shape,
        mode,
        cval=0.0,
        order=order,
        integer_output=False,
    )
    dx = cupy.zeros(ndim)

    # ret = _get_output(output, input, coordinates.shape[:-1])

    if sample_points.shape[0] != img.ndim:
        raise ValueError(
            "sample_points should have shape ndim on the first axis"
        )
    if not sample_points.flags.c_contiguous:
        sample_points = cupy.ascontiguousarray(sample_points)

    if coord_axis == 0:
        out_shape = (ndim,) + sample_points.shape[1:]
    elif coord_axis == -1:
        out_shape = sample_points.shape[1:] + (ndim,)
    else:
        raise ValueError("invalid coordinate axis")
    if out is None:
        out = cupy.empty(out_shape, dtype=img.dtype)
    elif out.shape != out_shape:
        raise ValueError("output array should have shape: {}".format(out_shape))

    tmp = cupy.empty(sample_points.shape[1:], dtype=img.dtype)
    inside = cupy.ones(tmp.shape, dtype=np.int32)
    for ax in range(img.ndim):
        dx[ax] = -0.5 * img_spacing[ax]
        grad_kernel(img, img_world2grid, sample_points, dx, tmp, inside)
        dx[ax] = 0.5 * img_spacing[ax]
        if coord_axis == 0:
            grad_kernel(img, img_world2grid, sample_points, dx, out[ax], inside)
            out[ax] -= tmp
            out[ax] /= img_spacing[ax]
        else:
            grad_kernel(
                img, img_world2grid, sample_points, dx, out[..., ax], inside
            )
            out[..., ax] -= tmp
            out[..., ax] /= img_spacing[ax]
        dx[ax] = 0
    return out, inside
