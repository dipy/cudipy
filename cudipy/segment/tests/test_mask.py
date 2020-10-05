import warnings

import numpy as np
import pytest
from numpy.testing import assert_equal

import cupy as cp
from cupy.testing import assert_array_equal
from cudipy.segment.mask import (
    applymask,
    bounding_box,
    crop,
    median_otsu,
    multi_median,
    otsu,
)
from cupyimg.scipy.ndimage import binary_dilation, generate_binary_structure
from cupyimg.scipy.ndimage.filters import median_filter
from dipy.data import get_fnames
from dipy.io.image import load_nifti_data


def test_mask():
    vol = cp.zeros((30, 30, 30))
    vol[15, 15, 15] = 1
    struct = generate_binary_structure(3, 1)
    # TODO: remove brute_force=True once non-brute force implemented for CuPy
    voln = binary_dilation(
        vol, structure=struct, iterations=4, brute_force=True
    ).astype("f4")
    initial = cp.sum(voln > 0)
    mask = voln.copy()
    thresh = otsu(mask)
    mask = mask > thresh
    initial_otsu = cp.sum(mask > 0)
    assert_array_equal(initial_otsu, initial)

    mins, maxs = bounding_box(mask)
    voln_crop = crop(mask, mins, maxs)
    initial_crop = cp.sum(voln_crop > 0)
    assert_array_equal(initial_crop, initial)

    applymask(voln, mask)
    final = cp.sum(voln > 0)
    assert_array_equal(final, initial)

    # Test multi_median.
    img = cp.arange(25).reshape(5, 5)
    img_copy = img.copy()
    medianradius = 2
    median_test = multi_median(img, medianradius, 3)
    assert_array_equal(img, img_copy)

    medarr = ((medianradius * 2) + 1,) * img.ndim
    median_control = median_filter(img, medarr)
    median_control = median_filter(median_control, medarr)
    median_control = median_filter(median_control, medarr)
    assert_array_equal(median_test, median_control)


def test_bounding_box():
    vol = cp.zeros((100, 100, 50), dtype=int)

    # Check the more usual case
    vol[10:90, 11:40, 5:33] = 3
    mins, maxs = bounding_box(vol)
    assert_equal(mins, [10, 11, 5])
    assert_equal(maxs, [90, 40, 33])

    # Check a 2d case
    mins, maxs = bounding_box(vol[10])
    assert_equal(mins, [11, 5])
    assert_equal(maxs, [40, 33])

    vol[:] = 0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Trigger a warning.
        num_warns = len(w)
        mins, maxs = bounding_box(vol)
        # Assert number of warnings has gone up by 1
        assert_equal(len(w), num_warns + 1)

        # Check that an empty array returns zeros for both min & max
        assert_equal(mins, [0, 0, 0])
        assert_equal(maxs, [0, 0, 0])

        # Check the 2d case
        mins, maxs = bounding_box(vol[0])
        assert_equal(len(w), num_warns + 2)
        assert_equal(mins, [0, 0])
        assert_equal(maxs, [0, 0])


def test_median_otsu():
    fname = get_fnames('S0_10')
    data = load_nifti_data(fname)
    data = cp.asarray(np.squeeze(data.astype('f8')))
    dummy_mask = data > data.mean()
    data_masked, mask = median_otsu(
        data,
        median_radius=3,
        numpass=2,
        autocrop=False,
        vol_idx=None,
        dilate=None,
    )
    assert mask.sum() < dummy_mask.sum()
    data2 = cp.zeros(data.shape + (2,))
    data2[..., 0] = data
    data2[..., 1] = data

    data2_masked, mask2 = median_otsu(
        data2,
        median_radius=3,
        numpass=2,
        autocrop=False,
        vol_idx=[0, 1],
        dilate=None,
    )
    assert mask.sum() == mask2.sum()

    _, mask3 = median_otsu(
        data2,
        median_radius=3,
        numpass=2,
        autocrop=False,
        vol_idx=[0, 1],
        dilate=1,
    )
    assert mask2.sum() < mask3.sum()

    _, mask4 = median_otsu(
        data2,
        median_radius=3,
        numpass=2,
        autocrop=False,
        vol_idx=[0, 1],
        dilate=2,
    )
    assert mask3.sum() < mask4.sum()

    # For 4D volumes, can't call without vol_idx input:
    with pytest.raises(ValueError):
        median_otsu(data2)
