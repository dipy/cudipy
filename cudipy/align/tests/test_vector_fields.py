import numpy as np
import pytest
from numpy.testing import assert_almost_equal
import cupy

from dipy.core import geometry
from dipy.align import floating
from dipy.align import imwarp
from dipy.align import vector_fields as vfu
from dipy.align.transforms import regtransforms
from dipy.align.parzenhist import sample_domain_regular


from cudipy.align import (
    compose_vector_fields,
    gradient,
    invert_vector_field_fixed_point,
    reorient_vector_field,
    sparse_gradient,
    transform_affine,
    warp,
)


pytest.importorskip("dipy")
pytest.importorskip("nibabel")


@pytest.mark.parametrize("shape", [(32, 48), (96, 64, 32), (64, 64, 64)])
def test_warp(shape):
    """Tests the cython implementation of the 3d warpings against scipy."""

    ndim = len(shape)
    radius = shape[0] / 3

    if ndim == 3:
        # Create an image of a sphere
        volume = vfu.create_sphere(*shape, radius)
        volume = np.array(volume, dtype=floating)

        # Create a displacement field for warping
        d, dinv = vfu.create_harmonic_fields_3d(*shape, 0.2, 8)
    else:
        # Create an image of a circle
        volume = vfu.create_circle(*shape, radius)
        volume = np.array(volume, dtype=floating)

        # Create a displacement field for warping
        d, dinv = vfu.create_harmonic_fields_2d(*shape, 0.2, 8)
    d = np.asarray(d).astype(floating)

    if ndim == 3:
        # Select an arbitrary rotation axis
        axis = np.array([0.5, 2.0, 1.5])
        # Select an arbitrary translation matrix
        t = 0.1
        trans = np.array(
            [
                [1, 0, 0, -t * shape[0]],
                [0, 1, 0, -t * shape[1]],
                [0, 0, 1, -t * shape[2]],
                [0, 0, 0, 1],
            ]
        )
        trans_inv = np.linalg.inv(trans)
        theta = np.pi / 5
        s = 1.1
        rot = np.zeros(shape=(4, 4))
        rot[:3, :3] = geometry.rodrigues_axis_rotation(axis, theta)
        rot[3, 3] = 1.0

        scale = np.array(
            [[1 * s, 0, 0, 0], [0, 1 * s, 0, 0], [0, 0, 1 * s, 0], [0, 0, 0, 1]]
        )
    elif ndim == 2:
        # Select an arbitrary translation matrix
        t = 0.1
        trans = np.array(
            [[1, 0, -t * shape[0]], [0, 1, -t * shape[1]], [0, 0, 1]]
        )
        trans_inv = np.linalg.inv(trans)
        theta = -1 * np.pi / 6.0
        s = 0.42
        ct = np.cos(theta)
        st = np.sin(theta)

        rot = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])

        scale = np.array([[1 * s, 0, 0], [0, 1 * s, 0], [0, 0, 1]])

    aff = trans_inv.dot(scale.dot(rot.dot(trans)))

    # Select arbitrary (but different) grid-to-space transforms
    sampling_grid2world = scale
    field_grid2world = aff
    field_world2grid = np.linalg.inv(field_grid2world)
    image_grid2world = aff.dot(scale)
    image_world2grid = np.linalg.inv(image_grid2world)

    A = field_world2grid.dot(sampling_grid2world)
    B = image_world2grid.dot(sampling_grid2world)
    C = image_world2grid

    # Reorient the displacement field according to its grid-to-space
    # transform
    dcopy = np.copy(d)
    if ndim == 3:
        vfu.reorient_vector_field_3d(dcopy, field_grid2world)
        expected = vfu.warp_3d(
            volume, dcopy, A, B, C, np.array(shape, dtype=np.int32)
        )
    elif ndim == 2:
        vfu.reorient_vector_field_2d(dcopy, field_grid2world)
        expected = vfu.warp_2d(
            volume, dcopy, A, B, C, np.array(shape, dtype=np.int32)
        )

    dcopyg = cupy.asarray(dcopy)
    volumeg = cupy.asarray(volume)
    Ag = cupy.asarray(A)
    Bg = cupy.asarray(B)
    Cg = cupy.asarray(C)

    warped = warp(volumeg, dcopyg, Ag, Bg, Cg, order=1, mode="grid-constant")

    cupy.testing.assert_array_almost_equal(warped, expected, decimal=4)


@pytest.mark.parametrize(
    "d_shape, codomain_shape, order",
    [
        [(64, 64), (80, 80), 1],
        [(64, 64), (80, 80), 0],
        [(64, 64, 64), (80, 80, 80), 1],
        [(64, 64, 64), (80, 80, 80), 0],
    ],
)
def test_transform_affine(d_shape, codomain_shape, order):

    ndim = len(d_shape)
    theta = -1 * np.pi / 5.0
    s = 0.5
    ct = np.cos(theta)
    st = np.sin(theta)

    if ndim == 2:
        # Create an image of a circle
        radius = d_shape[0] // 4
        volume = vfu.create_circle(*codomain_shape, radius)
        volume = np.array(volume, dtype=floating)

        # Generate affine transforms
        t = 0.3
        trans = np.array(
            [[1, 0, -t * d_shape[0]], [0, 1, -t * d_shape[1]], [0, 0, 1]]
        )
        trans_inv = np.linalg.inv(trans)
        rot = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])

        scale = np.array([[1 * s, 0, 0], [0, 1 * s, 0], [0, 0, 1]])
    elif ndim == 3:
        # Create an image of a sphere
        radius = d_shape[0] // 4
        volume = vfu.create_sphere(*codomain_shape, radius)
        volume = np.array(volume, dtype=floating)

        # Generate affine transforms
        # Select an arbitrary rotation axis
        axis = np.array([0.5, 2.0, 1.5])
        t = 0.3
        trans = np.array(
            [
                [1, 0, 0, -t * d_shape[0]],
                [0, 1, 0, -t * d_shape[1]],
                [0, 0, 1, -t * d_shape[2]],
                [0, 0, 0, 1],
            ]
        )
        trans_inv = np.linalg.inv(trans)

        rot = np.zeros(shape=(4, 4))
        rot[:3, :3] = geometry.rodrigues_axis_rotation(axis, theta)
        rot[3, 3] = 1.0

        scale = np.array(
            [[1 * s, 0, 0, 0], [0, 1 * s, 0, 0], [0, 0, 1 * s, 0], [0, 0, 0, 1]]
        )
    gt_affine = trans_inv.dot(scale.dot(rot.dot(trans)))

    # # Apply the affine transform to the grid coordinates
    # Y = np.apply_along_axis(gt_affine.dot, 0, X)[0:2, ...]

    # expected = map_coordinates(volume, Y, order=1)
    if order == 1:
        if ndim == 2:
            dipy_func = vfu.transform_2d_affine
        elif ndim == 3:
            dipy_func = vfu.transform_3d_affine
    elif order == 0:
        if ndim == 2:
            dipy_func = vfu.transform_2d_affine_nn
        elif ndim == 3:
            dipy_func = vfu.transform_3d_affine_nn
    expected = dipy_func(volume, np.array(d_shape, dtype=np.int32), gt_affine)

    volumed = cupy.asarray(volume)
    warped = transform_affine(volumed, d_shape, gt_affine, order=order)

    cupy.testing.assert_array_almost_equal(warped, expected)


@pytest.mark.parametrize("shape", [(10, 10), (10, 10, 10)])
def test_compose_vector_fields(shape):
    r"""
    Creates two random displacement field that exactly map pixels from an input
    image to an output image. The resulting displacements and their
    composition, although operating in physical space, map the points exactly
    (up to numerical precision).
    """
    np.random.seed(8315759)
    input_shape = shape
    tgt_sh = shape
    ndim = len(shape)
    if ndim == 3:
        # create a simple affine transformation
        ns = input_shape[0]
        nr = input_shape[1]
        nc = input_shape[2]
        s = 1.5
        t = 2.5
        trans = np.array(
            [
                [1, 0, 0, -t * ns],
                [0, 1, 0, -t * nr],
                [0, 0, 1, -t * nc],
                [0, 0, 0, 1],
            ]
        )
        trans_inv = np.linalg.inv(trans)
        scale = np.array(
            [[1 * s, 0, 0, 0], [0, 1 * s, 0, 0], [0, 0, 1 * s, 0], [0, 0, 0, 1]]
        )
        dipy_func = vfu.compose_vector_fields_3d
        dipy_create_func = vfu.create_random_displacement_3d
    elif ndim == 2:
        # create a simple affine transformation
        nr = input_shape[0]
        nc = input_shape[1]
        s = 1.5
        t = 2.5
        trans = np.array([[1, 0, -t * nr], [0, 1, -t * nc], [0, 0, 1]])
        trans_inv = np.linalg.inv(trans)
        scale = np.array([[1 * s, 0, 0], [0, 1 * s, 0], [0, 0, 1]])
        dipy_func = vfu.compose_vector_fields_2d
        dipy_create_func = vfu.create_random_displacement_2d

    gt_affine = trans_inv.dot(scale.dot(trans))

    # create two random displacement fields
    input_grid2world = gt_affine
    target_grid2world = gt_affine

    disp1, assign1 = dipy_create_func(
        np.array(input_shape, dtype=np.int32),
        input_grid2world,
        np.array(tgt_sh, dtype=np.int32),
        target_grid2world,
    )
    disp1 = np.array(disp1, dtype=floating)
    assign1 = np.array(assign1)

    disp2, assign2 = dipy_create_func(
        np.array(input_shape, dtype=np.int32),
        input_grid2world,
        np.array(tgt_sh, dtype=np.int32),
        target_grid2world,
    )
    disp2 = np.array(disp2, dtype=floating)
    assign2 = np.array(assign2)

    # create a random image (with decimal digits) to warp
    moving_image = np.empty(tgt_sh, dtype=floating)
    moving_image[...] = np.random.randint(0, 10, np.size(moving_image)).reshape(
        tuple(tgt_sh)
    )
    # set boundary values to zero so we don't test wrong interpolation due to
    # floating point precision
    if ndim == 3:
        moving_image[0, :, :] = 0
        moving_image[-1, :, :] = 0
        moving_image[:, 0, :] = 0
        moving_image[:, -1, :] = 0
        moving_image[:, :, 0] = 0
        moving_image[:, :, -1] = 0
        # evaluate the composed warping using the exact assignments
        # (first 1 then 2)

        warp1 = moving_image[
            (assign2[..., 0], assign2[..., 1], assign2[..., 2])
        ]
        expected = warp1[(assign1[..., 0], assign1[..., 1], assign1[..., 2])]

    elif ndim == 2:
        moving_image[0, :] = 0
        moving_image[-1, :] = 0
        moving_image[:, 0] = 0
        moving_image[:, -1] = 0
        # evaluate the composed warping using the exact assignments
        # (first 1 then 2)

        warp1 = moving_image[(assign2[..., 0], assign2[..., 1])]
        expected = warp1[(assign1[..., 0], assign1[..., 1])]

    # compose the displacement fields
    target_world2grid = np.linalg.inv(target_grid2world)
    premult_index = target_world2grid.dot(input_grid2world)
    premult_disp = target_world2grid

    disp1d = cupy.asarray(disp1)
    disp2d = cupy.asarray(disp2)
    premult_indexd = cupy.asarray(premult_index)
    premult_dispd = cupy.asarray(premult_disp)
    moving_imaged = cupy.asarray(moving_image)

    for time_scaling in [0.25, 1.0, 4.0]:
        composition, stats = dipy_func(
            disp1,
            disp2 / time_scaling,
            premult_index,
            premult_disp,
            time_scaling,
            None,
        )
        compositiond, statsd = compose_vector_fields(
            disp1d,
            disp2d / time_scaling,
            premult_indexd,
            premult_dispd,
            time_scaling,
            None,
        )
        cupy.testing.assert_array_almost_equal(composition, compositiond)
        cupy.testing.assert_array_almost_equal(stats, statsd)

        for order in [0, 1]:
            warped = warp(
                moving_imaged,
                compositiond,
                None,
                premult_indexd,
                premult_dispd,
                order=order,
            )
            cupy.testing.assert_array_almost_equal(warped, expected)

        # test updating the displacement field instead of creating a new one
        compositiond = disp1d.copy()
        compose_vector_fields(
            compositiond,
            disp2d / time_scaling,
            premult_indexd,
            premult_dispd,
            time_scaling,
            compositiond,
        )

        for order in [0, 1]:
            warped = warp(
                moving_imaged,
                compositiond,
                None,
                premult_indexd,
                premult_dispd,
                order=order,
            )
            cupy.testing.assert_array_almost_equal(warped, expected)

    # Test non-overlapping case
    if ndim == 3:
        x_0 = np.asarray(range(input_shape[0]))
        x_1 = np.asarray(range(input_shape[1]))
        x_2 = np.asarray(range(input_shape[2]))
        X = np.empty(input_shape + (3,), dtype=np.float64)
        O = np.ones(input_shape)
        X[..., 0] = x_0[:, None, None] * O
        X[..., 1] = x_1[None, :, None] * O
        X[..., 2] = x_2[None, None, :] * O
        sz = input_shape[0] * input_shape[1] * input_shape[2] * 3
        random_labels = np.random.randint(0, 2, sz)
        random_labels = random_labels.reshape(input_shape + (3,))
    elif ndim == 2:
        # Test non-overlapping case
        x_0 = np.asarray(range(input_shape[0]))
        x_1 = np.asarray(range(input_shape[1]))
        X = np.empty(input_shape + (2,), dtype=np.float64)
        O = np.ones(input_shape)
        X[..., 0] = x_0[:, None] * O
        X[..., 1] = x_1[None, :] * O
        random_labels = np.random.randint(
            0, 2, input_shape[0] * input_shape[1] * 2
        )
        random_labels = random_labels.reshape(input_shape + (2,))
    values = np.array([-1, tgt_sh[0]])
    disp1 = (values[random_labels] - X).astype(floating)
    disp1d = cupy.asarray(disp1)
    disp2d = cupy.asarray(disp2)
    composition, stats = compose_vector_fields(
        disp1d, disp2d, None, None, 1.0, None
    )
    cupy.testing.assert_array_almost_equal(
        composition, cupy.zeros_like(composition)
    )

    # test updating the displacement field instead of creating a new one
    compositiond = disp1d.copy()
    compose_vector_fields(compositiond, disp2d, None, None, 1.0, compositiond)
    cupy.testing.assert_array_almost_equal(
        compositiond, cupy.zeros_like(composition)
    )

    # TODO: resolve difference with DiPy by raising an error for the commented
    #       cases below? Currently second array allows 3x3
    # Test exception is raised when the affine transform matrix is not valid
    # if ndim == 3:
    #     valid = cupy.zeros((3, 4), dtype=cupy.float64)
    #     invalid = cupy.zeros((3, 3), dtype=cupy.float64)
    # elif ndim == 2:
    #     valid = cupy.zeros((2, 3), dtype=cupy.float64)
    #     invalid = cupy.zeros((2, 2), dtype=cupy.float64)
    # with pytest.raises(ValueError):
    #     compose_vector_fields(disp1d, disp2d, invalid, valid, 1.0, None)
    # with pytest.raises(ValueError):
    #     compose_vector_fields(disp1d, disp2d, valid, invalid, 1.0, None)


@pytest.mark.parametrize("shape", [(64, 64), (64, 64, 64)])
def test_invert_vector_field(shape):
    r"""
    Inverts a synthetic, analytically invertible, displacement field
    """
    ndim = len(shape)
    if ndim == 3:
        ns = shape[0]
        nr = shape[1]
        nc = shape[2]

        # Create an arbitrary image-to-space transform

        # Select an arbitrary rotation axis
        axis = np.array([2.0, 0.5, 1.0])
        t = 2.5  # translation factor

        trans = np.array(
            [
                [1, 0, 0, -t * ns],
                [0, 1, 0, -t * nr],
                [0, 0, 1, -t * nc],
                [0, 0, 0, 1],
            ]
        )
        dipy_create_func = vfu.create_harmonic_fields_3d
        dipy_reorient_func = vfu.reorient_vector_field_3d
        dipy_invert_func = vfu.invert_vector_field_fixed_point_3d
    elif ndim == 2:
        nr = shape[0]
        nc = shape[1]
        # Create an arbitrary image-to-space transform
        t = 2.5  # translation factor

        trans = np.array([[1, 0, -t * nr], [0, 1, -t * nc], [0, 0, 1]])
        dipy_create_func = vfu.create_harmonic_fields_2d
        dipy_reorient_func = vfu.reorient_vector_field_2d
        dipy_invert_func = vfu.invert_vector_field_fixed_point_2d

    trans_inv = np.linalg.inv(trans)

    d, _ = dipy_create_func(*shape, 0.2, 8)
    d = np.asarray(d).astype(floating)

    for theta in [-1 * np.pi / 5.0, 0.0, np.pi / 5.0]:  # rotation angle
        for s in [0.5, 1.0, 2.0]:  # scale
            if ndim == 3:
                rot = np.zeros(shape=(4, 4))
                rot[:3, :3] = geometry.rodrigues_axis_rotation(axis, theta)
                rot[3, 3] = 1.0
                scale = np.array(
                    [
                        [1 * s, 0, 0, 0],
                        [0, 1 * s, 0, 0],
                        [0, 0, 1 * s, 0],
                        [0, 0, 0, 1],
                    ]
                )
            elif ndim == 2:
                ct = np.cos(theta)
                st = np.sin(theta)

                rot = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])

                scale = np.array([[1 * s, 0, 0], [0, 1 * s, 0], [0, 0, 1]])

            gt_affine = trans_inv.dot(scale.dot(rot.dot(trans)))
            gt_affine_inv = np.linalg.inv(gt_affine)
            dcopy = np.copy(d)

            dcopyd = cupy.asarray(dcopy)
            gt_affined = cupy.asarray(gt_affine)
            gt_affine_invd = cupy.asarray(gt_affine_inv)

            # make sure the field remains invertible after the re-mapping
            dipy_reorient_func(dcopy, gt_affine)

            # TODO: can't do in-place computation unless out= is supplied and
            #       dcopy has the dimensions axis first instead of last
            dcopyd = reorient_vector_field(dcopyd, gt_affined)
            cupy.testing.assert_array_almost_equal(dcopyd, dcopy, decimal=4)

            # Note: the spacings are used just to check convergence, so they
            # don't need to be very accurate. Here we are passing (0.5 * s) to
            # force the algorithm to make more iterations: in ANTS, there is a
            # hard-coded bound on the maximum residual, that's why we cannot
            # force more iteration by changing the parameters.
            # We will investigate this issue with more detail in the future.

            if False:
                from cupyx.time import repeat

                perf = repeat(
                    invert_vector_field_fixed_point,
                    (
                        dcopyd,
                        gt_affine_invd,
                        cupy.asarray([s, s, s]) * 0.5,
                        40,
                        1e-7,
                    ),
                    n_warmup=20,
                    n_repeat=80,
                )
                print(perf)
                perf = repeat(
                    dipy_invert_func,
                    (
                        dcopy,
                        gt_affine_inv,
                        np.asarray([s, s, s]) * 0.5,
                        40,
                        1e-7,
                    ),
                    n_warmup=0,
                    n_repeat=8,
                )
                print(perf)
            # if False:
            #     from pyvolplot import volshow
            #     from matplotlib import pyplot as plt
            #     inv_approx, q, norms, tmp1, tmp2, epsilon, maxlen = vfu.invert_vector_field_fixed_point_3d_debug(
            #         dcopy, gt_affine_inv, np.array([s, s, s]) * 0.5, max_iter=1, tol=1e-7
            #     )
            #     inv_approxd, qd, normsd, tmp1d, tmp2d, epsilond, maxlend = invert_vector_field_fixed_point(
            #         dcopyd, gt_affine_invd, cupy.asarray([s, s, s]) * 0.5, max_iter=1, tol=1e-7
            #     )
            inv_approxd = invert_vector_field_fixed_point(
                dcopyd, gt_affine_invd, cupy.asarray([s, s, s]) * 0.5, 40, 1e-7
            )

            if False:
                inv_approx = dipy_invert_func(
                    dcopy, gt_affine_inv, np.array([s, s, s]) * 0.5, 40, 1e-7
                )
                cupy.testing.assert_allclose(
                    inv_approx, inv_approxd, rtol=1e-2, atol=1e-2
                )

            # TODO: use GPU-based imwarp here once implemented
            mapping = imwarp.DiffeomorphicMap(ndim, shape, gt_affine)
            mapping.forward = dcopy
            mapping.backward = inv_approxd.get()
            residual, stats = mapping.compute_inversion_error()
            assert_almost_equal(stats[1], 0, decimal=3)
            assert_almost_equal(stats[2], 0, decimal=3)

    # # Test exception is raised when the affine transform matrix is not valid
    # invalid = cupy.zeros((3, 3), dtype=np.float64)
    # spacing = cupy.asarray([1.0, 1.0, 1.0])
    # with pytest.raises(ValueError):
    #     invert_vector_field_fixed_point(dcopyd, invalid, spacing, 40, 1e-7, None)


def test_gradient_2d():
    np.random.seed(3921116)
    sh = (25, 32)
    # Create grid coordinates
    x_0 = np.arange(sh[0])
    x_1 = np.arange(sh[1])
    X = np.empty(sh + (3,), dtype=np.float64)
    O = np.ones(sh)
    X[..., 0] = x_0[:, None] * O
    X[..., 1] = x_1[None, :] * O
    X[..., 2] = 1

    transform = regtransforms[("RIGID", 2)]
    theta = np.array([0.1, 5.0, 2.5])
    T = transform.param_to_matrix(theta)
    TX = X.dot(T.T)
    # Eval an arbitrary (known) function at TX
    # f(x, y) = ax^2 + bxy + cy^{2}
    # df/dx = 2ax + by
    # df/dy = 2cy + bx
    a = 2e-3
    b = 5e-3
    c = 7e-3
    img = (
        a * TX[..., 0] ** 2 + b * TX[..., 0] * TX[..., 1] + c * TX[..., 1] ** 2
    )
    img = img.astype(floating)
    # img is an image sampled at X with grid-to-space transform T

    # Test sparse gradient: choose some sample points (in space)
    sample = sample_domain_regular(20, np.array(sh, dtype=np.int32), T)
    sample = np.array(sample)
    # Compute the analytical gradient at all points
    expected = np.empty((sample.shape[0], 2), dtype=floating)
    expected[..., 0] = 2 * a * sample[:, 0] + b * sample[:, 1]
    expected[..., 1] = 2 * c * sample[:, 1] + b * sample[:, 0]
    # Get the numerical gradient with the implementation under test
    sp_to_grid = np.linalg.inv(T)
    img_spacing = np.ones(2)

    img_d = cupy.asarray(img)
    img_spacing_d = cupy.asarray(img_spacing)
    sp_to_grid_d = cupy.asarray(sp_to_grid)
    sample_d = cupy.asarray(sample)

    actual, inside = vfu.sparse_gradient(img, sp_to_grid, img_spacing, sample)
    actual_gpu, inside_gpu = sparse_gradient(
        img_d, sp_to_grid_d, img_spacing_d, sample_d.T
    )
    atol = rtol = 1e-5
    cupy.testing.assert_allclose(
        actual * inside[..., np.newaxis],
        actual_gpu * inside_gpu[..., np.newaxis],
        atol=atol,
        rtol=rtol,
    )
    cupy.testing.assert_array_equal(inside, inside_gpu)

    # TODO: verify exceptions
    # # Verify exception is raised when passing invalid affine or spacings
    # invalid_affine = np.eye(2)
    # invalid_spacings = np.ones(1)
    # assert_raises(ValueError, vfu.sparse_gradient, img, invalid_affine,
    #               img_spacing, sample)
    # assert_raises(ValueError, vfu.sparse_gradient, img, sp_to_grid,
    #               invalid_spacings, sample)

    # Test dense gradient
    # Compute the analytical gradient at all points
    expected = np.empty(sh + (2,), dtype=floating)
    expected[..., 0] = 2 * a * TX[..., 0] + b * TX[..., 1]
    expected[..., 1] = 2 * c * TX[..., 1] + b * TX[..., 0]
    # Get the numerical gradient with the implementation under test
    sp_to_grid = np.linalg.inv(T)
    img_spacing = np.ones(2)

    actual, inside = vfu.gradient(img, sp_to_grid, img_spacing, sh, T)
    sp_to_grid_d = cupy.asarray(sp_to_grid)
    img_spacing_d = cupy.asarray(img_spacing)
    T_d = cupy.asarray(T)
    actual_gpu, inside_gpu = gradient(
        img_d, sp_to_grid_d, img_spacing_d, sh, T_d
    )

    atol = rtol = 1e-5
    cupy.testing.assert_allclose(
        actual * inside[..., np.newaxis],
        actual_gpu * inside_gpu[..., np.newaxis],
        atol=atol,
        rtol=rtol,
    )
    cupy.testing.assert_array_equal(inside, inside_gpu)

    # In the dense case, we are evaluating at the exact points (sample points
    # are not slightly moved like in the sparse case) so we have more precision

    # TODO: verify exceptions
    # # Verify exception is raised when passing invalid affine or spacings
    # assert_raises(ValueError, vfu.gradient, img, invalid_affine, img_spacing,
    #               sh, T)
    # assert_raises(ValueError, vfu.gradient, img, sp_to_grid, img_spacing,
    #               sh, invalid_affine)
    # assert_raises(ValueError, vfu.gradient, img, sp_to_grid, invalid_spacings,
    #               sh, T)


def test_gradient_3d():
    np.random.seed(3921116)
    shape = (25, 32, 15)
    # Create grid coordinates
    x_0 = np.asarray(range(shape[0]))
    x_1 = np.asarray(range(shape[1]))
    x_2 = np.asarray(range(shape[2]))
    X = np.zeros(shape + (4,), dtype=np.float64)
    O = np.ones(shape)
    X[..., 0] = x_0[:, None, None] * O
    X[..., 1] = x_1[None, :, None] * O
    X[..., 2] = x_2[None, None, :] * O
    X[..., 3] = 1

    transform = regtransforms[("RIGID", 3)]
    theta = np.array([0.1, 0.05, 0.12, -12.0, -15.5, -7.2])
    T = transform.param_to_matrix(theta)

    TX = X.dot(T.T)
    # Eval an arbitrary (known) function at TX
    # f(x, y, z) = ax^2 + by^2 + cz^2 + dxy + exz + fyz
    # df/dx = 2ax + dy + ez
    # df/dy = 2by + dx + fz
    # df/dz = 2cz + ex + fy
    a, b, c = 2e-3, 3e-3, 1e-3
    d, e, f = 1e-3, 2e-3, 3e-3
    img = (
        a * TX[..., 0] ** 2
        + b * TX[..., 1] ** 2
        + c * TX[..., 2] ** 2
        + d * TX[..., 0] * TX[..., 1]
        + e * TX[..., 0] * TX[..., 2]
        + f * TX[..., 1] * TX[..., 2]
    )

    img = img.astype(floating)
    # Test sparse gradient: choose some sample points (in space)
    sample = sample_domain_regular(100, np.array(shape, dtype=np.int32), T)
    sample = np.array(sample)
    # Compute the analytical gradient at all points
    expected = np.empty((sample.shape[0], 3), dtype=floating)
    expected[..., 0] = (
        2 * a * sample[:, 0] + d * sample[:, 1] + e * sample[:, 2]
    )
    expected[..., 1] = (
        2 * b * sample[:, 1] + d * sample[:, 0] + f * sample[:, 2]
    )
    expected[..., 2] = (
        2 * c * sample[:, 2] + e * sample[:, 0] + f * sample[:, 1]
    )
    # Get the numerical gradient with the implementation under test
    sp_to_grid = np.linalg.inv(T)
    img_spacing = np.ones(3)
    actual, inside = vfu.sparse_gradient(img, sp_to_grid, img_spacing, sample)

    img_d = cupy.asarray(img)
    img_spacing_d = cupy.asarray(img_spacing)
    sp_to_grid_d = cupy.asarray(sp_to_grid)
    sample_d = cupy.asarray(sample)
    actual_gpu, inside_gpu = sparse_gradient(
        img_d, sp_to_grid_d, img_spacing_d, sample_d.T
    )
    atol = rtol = 1e-5
    cupy.testing.assert_allclose(
        actual * inside[..., np.newaxis],
        actual_gpu * inside_gpu[..., np.newaxis],
        atol=atol,
        rtol=rtol,
    )
    cupy.testing.assert_array_equal(inside, inside_gpu)

    # TODO: test invalid inputs
    # # Verify exception is raised when passing invalid affine or spacings
    # invalid_affine = np.eye(3)
    # invalid_spacings = np.ones(2)
    # assert_raises(ValueError, vfu.sparse_gradient, img, invalid_affine,
    #               img_spacing, sample)
    # assert_raises(ValueError, vfu.sparse_gradient, img, sp_to_grid,
    #               invalid_spacings, sample)

    # Test dense gradient
    # Compute the analytical gradient at all points
    expected = np.empty(shape + (3,), dtype=floating)
    expected[..., 0] = 2 * a * TX[..., 0] + d * TX[..., 1] + e * TX[..., 2]
    expected[..., 1] = 2 * b * TX[..., 1] + d * TX[..., 0] + f * TX[..., 2]
    expected[..., 2] = 2 * c * TX[..., 2] + e * TX[..., 0] + f * TX[..., 1]
    # Get the numerical gradient with the implementation under test
    sp_to_grid = np.linalg.inv(T)
    img_spacing = np.ones(3)
    actual, inside = vfu.gradient(img, sp_to_grid, img_spacing, shape, T)

    sp_to_grid_d = cupy.asarray(sp_to_grid)
    img_spacing_d = cupy.asarray(img_spacing)
    T_d = cupy.asarray(T)
    actual_gpu, inside_gpu = gradient(
        img_d, sp_to_grid_d, img_spacing_d, shape, T_d
    )

    atol = rtol = 1e-5
    cupy.testing.assert_allclose(
        actual * inside[..., np.newaxis],
        actual_gpu * inside_gpu[..., np.newaxis],
        atol=atol,
        rtol=rtol,
    )
    cupy.testing.assert_array_equal(inside, inside_gpu)

    # TODO: test invalid inputs
    # # In the dense case, we are evaluating at the exact points (sample points
    # # are not slightly moved like in the sparse case) so we have more precision
    # assert_equal(diff.max() < 1e-5, True)
    # # Verify exception is raised when passing invalid affine or spacings
    # assert_raises(ValueError, vfu.gradient, img, invalid_affine, img_spacing,
    #               shape, T)
    # assert_raises(ValueError, vfu.gradient, img, sp_to_grid, img_spacing,
    #               shape, invalid_affine)
    # assert_raises(ValueError, vfu.gradient, img, sp_to_grid, invalid_spacings,
    #               shape, T)
