from warnings import warn

import cupy as cp
from cupyx.scipy.ndimage.filters import median_filter
from cupyx.scipy.ndimage import binary_dilation, generate_binary_structure


from .._utils import get_array_module

_otsu_available = False

try:
    from cupyimg.skimage.filters import threshold_otsu as otsu
    _otsu_available = True
except ImportError:

    def otsu(*args, **kwargs):
        raise ImportError("cupyimg is required to use otsu")


def multi_median(input, median_radius, numpass):
    """ Applies median filter multiple times on input data.

    Parameters
    ----------
    input : ndarray
        The input volume to apply filter on.
    median_radius : int
        Radius (in voxels) of the applied median filter
    numpass: int
        Number of pass of the median filter

    Returns
    -------
    input : ndarray
        Filtered input volume.
    """
    # Array representing the size of the median window in each dimension.
    size = ((median_radius * 2) + 1,) * input.ndim

    if numpass > 1:
        # ensure the input array is not modified
        input = input.copy()

    xp = get_array_module(input)

    # Multi pass
    output = xp.empty_like(input)
    for i in range(0, numpass):
        median_filter(input, size, output=output)
        input, output = output, input
    return input


def applymask(vol, mask):
    """ Mask vol with mask.

    Parameters
    ----------
    vol : ndarray
        Array with $V$ dimensions
    mask : ndarray
        Binary mask.  Has $M$ dimensions where $M <= V$. When $M < V$, we
        append $V - M$ dimensions with axis length 1 to `mask` so that `mask`
        will broadcast against `vol`.  In the typical case `vol` can be 4D,
        `mask` can be 3D, and we append a 1 to the mask shape which (via numpy
        broadcasting) has the effect of appling the 3D mask to each 3D slice in
        `vol` (``vol[..., 0]`` to ``vol[..., -1``).

    Returns
    -------
    masked_vol : ndarray
        `vol` multiplied by `mask` where `mask` may have been extended to match
        extra dimensions in `vol`
    """
    mask = mask.reshape(mask.shape + (vol.ndim - mask.ndim) * (1,))
    return vol * mask


def bounding_box(vol):
    """Compute the bounding box of nonzero intensity voxels in the volume.

    Parameters
    ----------
    vol : ndarray
        Volume to compute bounding box on.

    Returns
    -------
    npmins : list
        Array containg minimum index of each dimension
    npmaxs : list
        Array containg maximum index of each dimension
    """
    # Find bounds on first dimension
    temp = vol
    for i in range(vol.ndim - 1):
        temp = temp.any(-1)
    mins = [int(temp.argmax())]
    maxs = [len(temp) - int(temp[::-1].argmax())]
    # Check that vol is not all 0
    if mins[0] == 0 and temp[0] == 0:
        warn('No data found in volume to bound. Returning empty bounding box.')
        return [0] * vol.ndim, [0] * vol.ndim
    # Find bounds on remaining dimensions
    if vol.ndim > 1:
        a, b = bounding_box(vol.any(0))
        mins.extend(a)
        maxs.extend(b)
    return mins, maxs


def crop(vol, mins, maxs):
    """Crops the input volume.

    Parameters
    ----------
    vol : ndarray
        Volume to crop.
    mins : array
        Array containg minimum index of each dimension.
    maxs : array
        Array containg maximum index of each dimension.

    Returns
    -------
    vol : ndarray
        The cropped volume.
    """
    return vol[tuple(slice(i, j) for i, j in zip(mins, maxs))]


def median_otsu(input_volume, vol_idx=None, median_radius=4, numpass=4,
                autocrop=False, dilate=None):
    """Simple brain extraction tool method for images from DWI data.

    It uses a median filter smoothing of the input_volumes `vol_idx` and an
    automatic histogram Otsu thresholding technique, hence the name
    *median_otsu*.

    This function is inspired from Mrtrix's bet which has default values
    ``median_radius=3``, ``numpass=2``. However, from tests on multiple 1.5T
    and 3T data     from GE, Philips, Siemens, the most robust choice is
    ``median_radius=4``, ``numpass=4``.

    Parameters
    ----------
    input_volume : ndarray
        3D or 4D array of the brain volume.
    vol_idx : None or array, optional.
        1D array representing indices of ``axis=3`` of a 4D `input_volume`.
        None is only an acceptable input if ``input_volume`` is 3D.
    median_radius : int
        Radius (in voxels) of the applied median filter (default: 4).
    numpass: int
        Number of pass of the median filter (default: 4).
    autocrop: bool, optional
        if True, the masked input_volume will also be cropped using the
        bounding box defined by the masked data. Should be on if DWI is
        upsampled to 1x1x1 resolution. (default: False).
    dilate : None or int, optional
        number of iterations for binary dilation

    Returns
    -------
    maskedvolume : ndarray
        Masked input_volume
    mask : 3D ndarray
        The binary brain mask

    Notes
    -----
    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
    """
    if not _otsu_available:
        raise ImportError("cupyimg is required to use median_otsu")
    xp = get_array_module(input_volume, vol_idx)
    if len(input_volume.shape) == 4:
        if vol_idx is not None:
            b0vol = xp.mean(input_volume[..., xp.asarray(vol_idx)], axis=3)
        else:
            raise ValueError("For 4D images, must provide vol_idx input")
    else:
        b0vol = input_volume
    # Make a mask using a multiple pass median filter and histogram
    # thresholding.
    mask = multi_median(b0vol, median_radius, numpass)
    thresh = otsu(mask)
    mask = mask > thresh

    if dilate is not None:
        cross = generate_binary_structure(3, 1)
        # only brute_force iterations have been implemented in cupy
        kwargs = dict(brute_force=True) if xp == cp else {}
        mask = binary_dilation(mask, cross, iterations=dilate, **kwargs)

    # Auto crop the volumes using the mask as input_volume for bounding box
    # computing.
    if autocrop:
        mins, maxs = bounding_box(mask)
        mask = crop(mask, mins, maxs)
        croppedvolume = crop(input_volume, mins, maxs)
        maskedvolume = applymask(croppedvolume, mask)
    else:
        maskedvolume = applymask(input_volume, mask)
    return maskedvolume, mask
