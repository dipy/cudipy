import numpy as np
import pytest

import cupy as cp
from cudipy.align import floating
from cudipy.align.crosscorr import (
    compute_cc_backward_step,
    compute_cc_forward_step,
    precompute_cc_factors,
)
from dipy.align import crosscorr as cc


@pytest.mark.parametrize("shape", [(20, 20), (20, 20, 20)])
def test_cc_factors(shape):
    r"""
    Compares the output of the optimized function to compute the cross-
    correlation factors against a direct (not optimized, but less error prone)
    implementation.
    """
    size = np.prod(shape)
    a = np.array(range(size), dtype=floating).reshape(shape)
    b = np.array(range(size)[::-1], dtype=floating).reshape(shape)
    a /= a.max()
    b /= b.max()
    ndim = len(shape)
    if ndim == 2:
        dipy_func = cc.precompute_cc_factors_2d_test
    else:
        dipy_func = cc.precompute_cc_factors_3d_test
    for radius in [0, 1, 3, 6]:
        expected = np.asarray(dipy_func(a, b, radius))
        factors = precompute_cc_factors(cp.asarray(a), cp.asarray(b), radius)
        factors = cp.stack(factors, axis=-1)
        cp.testing.assert_allclose(factors, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(32, 32), (32, 32, 32)])
def test_compute_cc_steps(shape):
    # Select arbitrary images' shape (same shape for both images)
    sh = shape
    ndim = len(sh)
    radius = 2

    # Select arbitrary centers
    c_f = (np.asarray(sh) / 2) + 1.25
    c_g = c_f + 2.5

    if ndim == 3:
        # Compute the identity vector field I(x) = x in R^2
        x_0 = np.asarray(range(sh[0]))
        x_1 = np.asarray(range(sh[1]))
        x_2 = np.asarray(range(sh[2]))
        X = np.ndarray(sh + (3,), dtype=np.float64)
        O = np.ones(sh)
        X[..., 0] = x_0[:, None, None] * O
        X[..., 1] = x_1[None, :, None] * O
        X[..., 2] = x_2[None, None, :] * O
        dipy_precompute = cc.precompute_cc_factors_3d_test
    elif ndim == 2:
        # Compute the identity vector field I(x) = x in R^2
        x_0 = np.asarray(range(sh[0]))
        x_1 = np.asarray(range(sh[1]))
        X = np.ndarray(sh + (2,), dtype=np.float64)
        O = np.ones(sh)
        X[..., 0] = x_0[:, None] * O
        X[..., 1] = x_1[None, :] * O
        dipy_precompute = cc.precompute_cc_factors_2d_test

    # Compute the gradient fields of F and G
    np.random.seed(1147572)

    gradF = np.array(X - c_f, dtype=floating)
    gradG = np.array(X - c_g, dtype=floating)

    sz = np.size(gradF)
    Fnoise = np.random.ranf(sz).reshape(gradF.shape) * gradF.max() * 0.1
    Fnoise = Fnoise.astype(floating)
    gradF += Fnoise

    sz = np.size(gradG)
    Gnoise = np.random.ranf(sz).reshape(gradG.shape) * gradG.max() * 0.1
    Gnoise = Gnoise.astype(floating)
    gradG += Gnoise

    sq_norm_grad_G = np.sum(gradG ** 2, -1)

    F = np.array(0.5 * np.sum(gradF ** 2, -1), dtype=floating)
    G = np.array(0.5 * sq_norm_grad_G, dtype=floating)

    Fnoise = np.random.ranf(np.size(F)).reshape(F.shape) * F.max() * 0.1
    Fnoise = Fnoise.astype(floating)
    F += Fnoise

    Gnoise = np.random.ranf(np.size(G)).reshape(G.shape) * G.max() * 0.1
    Gnoise = Gnoise.astype(floating)
    G += Gnoise

    # precompute the cross correlation factors
    factors = dipy_precompute(F, G, radius)
    factors = np.array(factors, dtype=floating)

    # test the forward step against the exact expression
    I = factors[..., 0]
    J = factors[..., 1]
    sfm = factors[..., 2]
    sff = factors[..., 3]
    smm = factors[..., 4]
    expected = np.ndarray(shape=sh + (ndim,), dtype=floating)
    factor = (-2.0 * sfm / (sff * smm)) * (J - (sfm / sff) * I)
    for n in range(ndim):
        expected[..., n] = factor * gradF[..., n]

    rtol = atol = 1e-6
    gradFd = cp.asarray(gradF)
    factorsd = [
        cp.asarray(I),
        cp.asarray(J),
        cp.asarray(sfm),
        cp.asarray(sff),
        cp.asarray(smm),
    ]
    actual, energy = compute_cc_forward_step(gradFd, factorsd, 0)
    cp.testing.assert_array_almost_equal(actual, expected)
    for radius in range(1, 5):
        expected[:radius, ...] = 0
        expected[:, :radius, ...] = 0
        expected[-radius:, ...] = 0
        expected[:, -radius:, ...] = 0
        if ndim == 3:
            expected[:, :, :radius, :] = 0
            expected[:, :, -radius:, ...] = 0
        actual, energy = compute_cc_forward_step(gradFd, factorsd, radius)
        cp.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)

    # test the backward step against the exact expression
    factor = (-2.0 * sfm / (sff * smm)) * (I - (sfm / smm) * J)
    for n in range(ndim):
        expected[..., n] = factor * gradG[..., n]
    gradGd = cp.asarray(gradG)
    actual, energy = compute_cc_backward_step(gradGd, factorsd, 0)
    cp.testing.assert_array_almost_equal(actual, expected)
    for radius in range(1, 5):
        expected[:radius, ...] = 0
        expected[:, :radius, ...] = 0
        expected[-radius:, ...] = 0
        expected[:, -radius:, ...] = 0
        if ndim == 3:
            expected[:, :, :radius, :] = 0
            expected[:, :, -radius:, ...] = 0
        actual, energy = compute_cc_backward_step(gradGd, factorsd, radius)
        cp.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
