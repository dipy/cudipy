""" Affine image registration module consisting of the following classes:

    AffineMap: encapsulates the necessary information to perform affine
        transforms between two domains, defined by a `static` and a `moving`
        image. The `domain` of the transform is the set of points in the
        `static` image's grid, and the `codomain` is the set of points in
        the `moving` image. When we call the `transform` method, `AffineMap`
        maps each point `x` of the domain (`static` grid) to the codomain
        (`moving` grid) and interpolates the `moving` image at that point
        to obtain the intensity value to be placed at `x` in the resulting
        grid. The `transform_inverse` method performs the opposite operation
        mapping points in the codomain to points in the domain.

    ParzenJointHistogram: computes the marginal and joint distributions of
        intensities of a pair of images, using Parzen windows [Parzen62]
        with a cubic spline kernel, as proposed by Mattes et al. [Mattes03].
        It also computes the gradient of the joint histogram w.r.t. the
        parameters of a given transform.

    MutualInformationMetric: computes the value and gradient of the mutual
        information metric the way `Optimizer` needs them. That is, given
        a set of transform parameters, it will use `ParzenJointHistogram`
        to compute the value and gradient of the joint intensity histogram
        evaluated at the given parameters, and evaluate the the value and
        gradient of the histogram's mutual information.

    AffineRegistration: it runs the multi-resolution registration, putting
        all the pieces together. It needs to create the scale space of the
        images and run the multi-resolution registration by using the Metric
        and the Optimizer at each level of the Gaussian pyramid. At each
        level, it will setup the metric to compute value and gradient of the
        metric with the input images with different levels of smoothing.

    References
    ----------
    [Parzen62] E. Parzen. On the estimation of a probability density
               function and the mode. Annals of Mathematical Statistics,
               33(3), 1065-1076, 1962.
    [Mattes03] Mattes, D., Haynor, D. R., Vesselle, H., Lewellen, T. K.,
               & Eubank, W. PET-CT image registration in the chest using
               free-form deformations. IEEE Transactions on Medical
               Imaging, 22(1), 120-8, 2003.

"""
import functools
# from warnings import warn

import cupy as cp
import numpy as np
from numpy import linalg
import cupyx.scipy.ndimage as ndimage

# from dipy.align import VerbosityLevels
# from dipy.align.parzenhist import (
#     ParzenJointHistogram,
#     compute_parzen_mi,
#     sample_domain_regular,
# )
from cudipy.align import vector_fields as vf
# from cudipy.align.imwarp import ScaleSpace, get_direction_and_spacings
# from cudipy.align.scalespace import IsotropicScaleSpace
# from dipy.core.interpolation import interpolate_scalar_2d, interpolate_scalar_3d
# from dipy.core.optimize import Optimizer
from dipy.utils.deprecator import deprecated_params

partial = functools.partial
_interp_options = ['nearest', 'linear']
_transform_method = {}
_transform_method[(2, 'nearest')] = partial(vf.transform_affine, order=0)
_transform_method[(3, 'nearest')] = partial(vf.transform_affine, order=0)
_transform_method[(2, 'linear')] = partial(vf.transform_affine, order=1)
_transform_method[(3, 'linear')] = partial(vf.transform_affine, order=1)
_number_dim_affine_matrix = 2


class AffineInversionError(Exception):
    pass


class AffineInvalidValuesError(Exception):
    pass


class AffineMap(object):

    def __init__(self, affine, domain_grid_shape=None, domain_grid2world=None,
                 codomain_grid_shape=None, codomain_grid2world=None):
        """ AffineMap

        Implements an affine transformation whose domain is given by
        `domain_grid` and `domain_grid2world`, and whose co-domain is
        given by `codomain_grid` and `codomain_grid2world`.

        The actual transform is represented by the `affine` matrix, which
        operate in world coordinates. Therefore, to transform a moving image
        towards a static image, we first map each voxel (i,j,k) of the static
        image to world coordinates (x,y,z) by applying `domain_grid2world`.
        Then we apply the `affine` transform to (x,y,z) obtaining (x', y', z')
        in moving image's world coordinates. Finally, (x', y', z') is mapped
        to voxel coordinates (i', j', k') in the moving image by multiplying
        (x', y', z') by the inverse of `codomain_grid2world`. The
        `codomain_grid_shape` is used analogously to transform the static
        image towards the moving image when calling `transform_inverse`.

        If the domain/co-domain information is not provided (None) then the
        sampling information needs to be specified each time the `transform`
        or `transform_inverse` is called to transform images. Note that such
        sampling information is not necessary to transform points defined in
        physical space, such as stream lines.

        Parameters
        ----------
        affine : array, shape (dim + 1, dim + 1)
            the matrix defining the affine transform, where `dim` is the
            dimension of the space this map operates in (2 for 2D images,
            3 for 3D images). If None, then `self` represents the identity
            transformation.
        domain_grid_shape : sequence, shape (dim,), optional
            the shape of the default domain sampling grid. When `transform`
            is called to transform an image, the resulting image will have
            this shape, unless a different sampling information is provided.
            If None, then the sampling grid shape must be specified each time
            the `transform` method is called.
        domain_grid2world : array, shape (dim + 1, dim + 1), optional
            the grid-to-world transform associated with the domain grid.
            If None (the default), then the grid-to-world transform is assumed
            to be the identity.
        codomain_grid_shape : sequence of integers, shape (dim,)
            the shape of the default co-domain sampling grid. When
            `transform_inverse` is called to transform an image, the resulting
            image will have this shape, unless a different sampling
            information is provided. If None (the default), then the sampling
            grid shape must be specified each time the `transform_inverse`
            method is called.
        codomain_grid2world : array, shape (dim + 1, dim + 1)
            the grid-to-world transform associated with the co-domain grid.
            If None (the default), then the grid-to-world transform is assumed
            to be the identity.

        """
        self.set_affine(affine)
        self.domain_shape = domain_grid_shape
        self.domain_grid2world = domain_grid2world
        self.codomain_shape = codomain_grid_shape
        self.codomain_grid2world = codomain_grid2world

    def get_affine(self):
        """Return the value of the transformation, not a reference.

        Returns
        -------
        affine : ndarray
            Copy of the transform, not a reference.

        """

        # returning a copy to insulate it from changes outside object
        return self.affine.copy()

    def set_affine(self, affine):
        """Set the affine transform (operating in physical space).

        Also sets `self.affine_inv` - the inverse of `affine`, or None if
        there is no inverse.

        Parameters
        ----------
        affine : array, shape (dim + 1, dim + 1)
            the matrix representing the affine transform operating in
            physical space. The domain and co-domain information
            remains unchanged. If None, then `self` represents the identity
            transformation.

        """

        if affine is None:
            self.affine = None
            self.affine_inv = None
            return

        try:
            affine = np.array(affine)
        except Exception:
            raise TypeError("Input must be type ndarray, or be convertible"
                            " to one.")

        if len(affine.shape) != _number_dim_affine_matrix:
            raise AffineInversionError('Affine transform must be 2D')

        if not affine.shape[0] == affine.shape[1]:
            raise AffineInversionError("Affine transform must be a square "
                                       "matrix")

        if isinstance(affine, cp.ndarray):
            # keep the affine matrix on the host
            affine = cp.asnumpy(affine)

        if not np.all(np.isfinite(affine)):
            raise AffineInvalidValuesError("Affine transform contains invalid"
                                           " elements")

        # checking on proper augmentation
        # First n-1 columns in last row in matrix contain non-zeros
        if not np.all(affine[-1, :-1] == 0.0):
            raise AffineInvalidValuesError("First {n_1} columns in last row"
                                           " in matrix contain non-zeros!"
                                           .format(n_1=affine.shape[0] - 1))

        # Last row, last column in matrix must be 1.0!
        if affine[-1, -1] != 1.0:
            raise AffineInvalidValuesError("Last row, last column in matrix"
                                           " is not 1.0!")

        # making a copy to insulate it from changes outside object
        self.affine = affine.copy()

        try:
            self.affine_inv = linalg.inv(affine)
        except linalg.LinAlgError:
            raise AffineInversionError('Affine cannot be inverted')

    def __str__(self):
        """Printable format - relies on ndarray's implementation."""

        return str(self.affine)

    def __repr__(self):
        """Relodable representation - also relies on ndarray's implementation."""

        return self.affine.__repr__()

    def __format__(self, format_spec):
        """Implementation various formatting options"""

        if format_spec is None or self.affine is None:
            return str(self.affine)
        elif isinstance(format_spec, str):
            format_spec = format_spec.lower()
            if format_spec in ['', ' ', 'f', 'full']:
                return str(self.affine)
            # rotation part only (initial 3x3)
            elif format_spec in ['r', 'rotation']:
                return str(self.affine[:-1, :-1])
            # translation part only (4th col)
            elif format_spec in ['t', 'translation']:
                # notice unusual indexing to make it a column vector
                #   i.e. rows from 0 to n-1, cols from n to n
                return str(self.affine[:-1, -1:])
            else:
                allowed_formats_print_map = ['full', 'f',
                                             'rotation', 'r',
                                             'translation', 't']
                raise NotImplementedError("Format {} not recognized or"
                                          "implemented.\nTry one of {}"
                                          .format(format_spec,
                                                  allowed_formats_print_map))

    @deprecated_params('interp', 'interpolation', since='1.13', until='1.15')
    def _apply_transform(self, image, interpolation='linear',
                         image_grid2world=None, sampling_grid_shape=None,
                         sampling_grid2world=None, resample_only=False,
                         apply_inverse=False):
        """Transform the input image applying this affine transform.

        This is a generic function to transform images using either this
        (direct) transform or its inverse.

        If applying the direct transform (`apply_inverse=False`):
            by default, the transformed image is sampled at a grid defined by
            `self.domain_shape` and `self.domain_grid2world`.
        If applying the inverse transform (`apply_inverse=True`):
            by default, the transformed image is sampled at a grid defined by
            `self.codomain_shape` and `self.codomain_grid2world`.

        If the sampling information was not provided at initialization of this
        transform then `sampling_grid_shape` is mandatory.

        Parameters
        ----------
        image :  2D or 3D array
            the image to be transformed
        interpolation : string, either 'linear' or 'nearest'
            the type of interpolation to be used, either 'linear'
            (for k-linear interpolation) or 'nearest' for nearest neighbor
        image_grid2world : array, shape (dim + 1, dim + 1), optional
            the grid-to-world transform associated with `image`.
            If None (the default), then the grid-to-world transform is assumed
            to be the identity.
        sampling_grid_shape : sequence, shape (dim,), optional
            the shape of the grid where the transformed image must be sampled.
            If None (the default), then `self.domain_shape` is used instead
            (which must have been set at initialization, otherwise an exception
            will be raised).
        sampling_grid2world : array, shape (dim + 1, dim + 1), optional
            the grid-to-world transform associated with the sampling grid
            (specified by `sampling_grid_shape`, or by default
            `self.domain_shape`). If None (the default), then the
            grid-to-world transform is assumed to be the identity.
        resample_only : Boolean, optional
            If False (the default) the affine transform is applied normally.
            If True, then the affine transform is not applied, and the input
            image is just re-sampled on the domain grid of this transform.
        apply_inverse : Boolean, optional
            If False (the default) the image is transformed from the codomain
            of this transform to its domain using the (direct) affine
            transform. Otherwise, the image is transformed from the domain
            of this transform to its codomain using the (inverse) affine
            transform.

        Returns
        -------
        transformed : array, shape `sampling_grid_shape` or `self.domain_shape`
            the transformed image, sampled at the requested grid

        """
        # Verify valid interpolation requested
        if interpolation not in _interp_options:
            msg = 'Unknown interpolation method: %s' % (interpolation,)
            raise ValueError(msg)

        # Obtain sampling grid
        if sampling_grid_shape is None:
            if apply_inverse:
                sampling_grid_shape = self.codomain_shape
            else:
                sampling_grid_shape = self.domain_shape
        if sampling_grid_shape is None:
            msg = 'Unknown sampling info. Provide a valid sampling_grid_shape'
            raise ValueError(msg)

        dim = len(sampling_grid_shape)
        shape = tuple(sampling_grid_shape)

        # Verify valid image dimension
        img_dim = len(image.shape)
        if img_dim < 2 or img_dim > 3:
            raise ValueError('Undefined transform for dim: %d' % (img_dim,))

        # Obtain grid-to-world transform for sampling grid
        if sampling_grid2world is None:
            if apply_inverse:
                sampling_grid2world = self.codomain_grid2world
            else:
                sampling_grid2world = self.domain_grid2world
        if sampling_grid2world is None:
            sampling_grid2world = np.eye(dim + 1)
        if isinstance(sampling_grid2world, cp.ndarray):
            sampling_grid2world = cp.asnumpy(sampling_grid2world)

        # Obtain world-to-grid transform for input image
        if image_grid2world is None:
            if apply_inverse:
                image_grid2world = self.domain_grid2world
            else:
                image_grid2world = self.codomain_grid2world
            if image_grid2world is None:
                image_grid2world = np.eye(dim + 1)
        image_world2grid = linalg.inv(image_grid2world)
        if isinstance(image_world2grid, cp.ndarray):
            image_world2grid = cp.asnumpy(image_world2grid)

        # Compute the transform from sampling grid to input image grid
        if apply_inverse:
            aff = self.affine_inv
        else:
            aff = self.affine

        if (aff is None) or resample_only:
            comp = image_world2grid.dot(sampling_grid2world)
        else:
            comp = image_world2grid.dot(aff.dot(sampling_grid2world))
        comp = cp.asarray(comp)

        # Transform the input image
        if interpolation == 'linear':
            image = image.astype(cp.promote_types(image.dtype, np.float32))

        transformed = _transform_method[(dim, interpolation)](image, shape,
                                                              comp)
        return transformed

    @deprecated_params('interp', 'interpolation', since='1.13', until='1.15')
    def transform(self, image, interpolation='linear', image_grid2world=None,
                  sampling_grid_shape=None, sampling_grid2world=None,
                  resample_only=False):
        """Transform the input image from co-domain to domain space.

        By default, the transformed image is sampled at a grid defined by
        `self.domain_shape` and `self.domain_grid2world`. If such
        information was not provided then `sampling_grid_shape` is mandatory.

        Parameters
        ----------
        image :  2D or 3D array
            the image to be transformed
        interpolation : string, either 'linear' or 'nearest'
            the type of interpolation to be used, either 'linear'
            (for k-linear interpolation) or 'nearest' for nearest neighbor
        image_grid2world : array, shape (dim + 1, dim + 1), optional
            the grid-to-world transform associated with `image`.
            If None (the default), then the grid-to-world transform is assumed
            to be the identity.
        sampling_grid_shape : sequence, shape (dim,), optional
            the shape of the grid where the transformed image must be sampled.
            If None (the default), then `self.codomain_shape` is used instead
            (which must have been set at initialization, otherwise an exception
            will be raised).
        sampling_grid2world : array, shape (dim + 1, dim + 1), optional
            the grid-to-world transform associated with the sampling grid
            (specified by `sampling_grid_shape`, or by default
            `self.codomain_shape`). If None (the default), then the
            grid-to-world transform is assumed to be the identity.
        resample_only : Boolean, optional
            If False (the default) the affine transform is applied normally.
            If True, then the affine transform is not applied, and the input
            image is just re-sampled on the domain grid of this transform.

        Returns
        -------
        transformed : array, shape `sampling_grid_shape` or
                      `self.codomain_shape`
            the transformed image, sampled at the requested grid

        """
        transformed = self._apply_transform(image, interpolation,
                                            image_grid2world,
                                            sampling_grid_shape,
                                            sampling_grid2world,
                                            resample_only,
                                            apply_inverse=False)
        return transformed

    @deprecated_params('interp', 'interpolation', since='1.13', until='1.15')
    def transform_inverse(self, image, interpolation='linear',
                          image_grid2world=None, sampling_grid_shape=None,
                          sampling_grid2world=None, resample_only=False):
        """Transform the input image from domain to co-domain space.

        By default, the transformed image is sampled at a grid defined by
        `self.codomain_shape` and `self.codomain_grid2world`. If such
        information was not provided then `sampling_grid_shape` is mandatory.

        Parameters
        ----------
        image :  2D or 3D array
            the image to be transformed
        interpolation : string, either 'linear' or 'nearest'
            the type of interpolation to be used, either 'linear'
            (for k-linear interpolation) or 'nearest' for nearest neighbor
        image_grid2world : array, shape (dim + 1, dim + 1), optional
            the grid-to-world transform associated with `image`.
            If None (the default), then the grid-to-world transform is assumed
            to be the identity.
        sampling_grid_shape : sequence, shape (dim,), optional
            the shape of the grid where the transformed image must be sampled.
            If None (the default), then `self.codomain_shape` is used instead
            (which must have been set at initialization, otherwise an exception
            will be raised).
        sampling_grid2world : array, shape (dim + 1, dim + 1), optional
            the grid-to-world transform associated with the sampling grid
            (specified by `sampling_grid_shape`, or by default
            `self.codomain_shape`). If None (the default), then the
            grid-to-world transform is assumed to be the identity.
        resample_only : Boolean, optional
            If False (the default) the affine transform is applied normally.
            If True, then the affine transform is not applied, and the input
            image is just re-sampled on the domain grid of this transform.

        Returns
        -------
        transformed : array, shape `sampling_grid_shape` or
                      `self.codomain_shape`
            the transformed image, sampled at the requested grid

        """
        transformed = self._apply_transform(image, interpolation,
                                            image_grid2world,
                                            sampling_grid_shape,
                                            sampling_grid2world,
                                            resample_only,
                                            apply_inverse=True)
        return transformed


def transform_centers_of_mass(static, static_grid2world,
                              moving, moving_grid2world):
    r""" Transformation to align the center of mass of the input images.

    Parameters
    ----------
    static : array, shape (S, R, C)
        static image
    static_grid2world : array, shape (dim+1, dim+1)
        the voxel-to-space transformation of the static image
    moving : array, shape (S, R, C)
        moving image
    moving_grid2world : array, shape (dim+1, dim+1)
        the voxel-to-space transformation of the moving image

    Returns
    -------
    affine_map : instance of AffineMap
        the affine transformation (translation only, in this case) aligning
        the center of mass of the moving image towards the one of the static
        image

    """
    dim = len(static.shape)
    if static_grid2world is None:
        static_grid2world = np.eye(dim + 1)
    if moving_grid2world is None:
        moving_grid2world = np.eye(dim + 1)
    c_static = ndimage.measurements.center_of_mass(np.array(static))
    c_static = static_grid2world.dot(c_static + (1,))
    c_moving = ndimage.measurements.center_of_mass(np.array(moving))
    c_moving = moving_grid2world.dot(c_moving + (1,))
    transform = np.eye(dim + 1)
    transform[:dim, dim] = (c_moving - c_static)[:dim]
    affine_map = AffineMap(transform,
                           static.shape, static_grid2world,
                           moving.shape, moving_grid2world)
    return affine_map


def transform_geometric_centers(static, static_grid2world,
                                moving, moving_grid2world):
    r""" Transformation to align the geometric center of the input images.

    With "geometric center" of a volume we mean the physical coordinates of
    its central voxel

    Parameters
    ----------
    static : array, shape (S, R, C)
        static image
    static_grid2world : array, shape (dim+1, dim+1)
        the voxel-to-space transformation of the static image
    moving : array, shape (S, R, C)
        moving image
    moving_grid2world : array, shape (dim+1, dim+1)
        the voxel-to-space transformation of the moving image

    Returns
    -------
    affine_map : instance of AffineMap
        the affine transformation (translation only, in this case) aligning
        the geometric center of the moving image towards the one of the static
        image

    """
    dim = len(static.shape)
    if static_grid2world is None:
        static_grid2world = np.eye(dim + 1)
    if moving_grid2world is None:
        moving_grid2world = np.eye(dim + 1)
    c_static = tuple((np.array(static.shape, dtype=np.float64)) * 0.5)
    c_static = static_grid2world.dot(c_static + (1,))
    c_moving = tuple((np.array(moving.shape, dtype=np.float64)) * 0.5)
    c_moving = moving_grid2world.dot(c_moving + (1,))
    transform = np.eye(dim + 1)
    transform[:dim, dim] = (c_moving - c_static)[:dim]
    affine_map = AffineMap(transform,
                           static.shape, static_grid2world,
                           moving.shape, moving_grid2world)
    return affine_map


def transform_origins(static, static_grid2world,
                      moving, moving_grid2world):
    r""" Transformation to align the origins of the input images.

    With "origin" of a volume we mean the physical coordinates of
    voxel (0,0,0)

    Parameters
    ----------
    static : array, shape (S, R, C)
        static image
    static_grid2world : array, shape (dim+1, dim+1)
        the voxel-to-space transformation of the static image
    moving : array, shape (S, R, C)
        moving image
    moving_grid2world : array, shape (dim+1, dim+1)
        the voxel-to-space transformation of the moving image

    Returns
    -------
    affine_map : instance of AffineMap
        the affine transformation (translation only, in this case) aligning
        the origin of the moving image towards the one of the static
        image

    """
    dim = len(static.shape)
    if static_grid2world is None:
        static_grid2world = np.eye(dim + 1)
    if moving_grid2world is None:
        moving_grid2world = np.eye(dim + 1)
    c_static = static_grid2world[:dim, dim]
    c_moving = moving_grid2world[:dim, dim]
    transform = np.eye(dim + 1)
    transform[:dim, dim] = (c_moving - c_static)[:dim]
    affine_map = AffineMap(transform,
                           static.shape, static_grid2world,
                           moving.shape, moving_grid2world)
    return affine_map


# TODO:
#     classes: MutualInformationMetric, AffineRegistration
