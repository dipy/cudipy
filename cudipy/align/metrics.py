"""  Metrics for Symmetric Diffeomorphic Registration """

import abc

import cupy as cp

# from cupyimg.dipy.align import sumsqdiff as ssd
from cudipy.align import crosscorr as cc
from cudipy.align import floating
from cudipy.align import vector_fields as vfu
from cupyx.scipy import ndimage

if hasattr(cp, 'gradient'):
    gradient = cp.gradient
else:
    # older CuPy does not have gradient implemented
    from cupyimg.numpy import gradient

# from cudipy.align import expectmax as em


class SimilarityMetric(object, metaclass=abc.ABCMeta):
    def __init__(self, dim):
        r""" Similarity Metric abstract class

        A similarity metric is in charge of keeping track of the numerical
        value of the similarity (or distance) between the two given images. It
        also computes the update field for the forward and inverse displacement
        fields to be used in a gradient-based optimization algorithm. Note that
        this metric does not depend on any transformation (affine or
        non-linear) so it assumes the static and moving images are already
        warped

        Parameters
        ----------
        dim : int (either 2 or 3)
            the dimension of the image domain
        """
        self.dim = dim
        self.levels_above = None
        self.levels_below = None

        self.static_image = None
        self.static_affine = None
        self.static_spacing = None
        self.static_direction = None

        self.moving_image = None
        self.moving_affine = None
        self.moving_spacing = None
        self.moving_direction = None
        self.mask0 = False

    def set_levels_below(self, levels):
        r"""Informs the metric how many pyramid levels are below the current one

        Informs this metric the number of pyramid levels below the current one.
        The metric may change its behavior (e.g. number of inner iterations)
        accordingly

        Parameters
        ----------
        levels : int
            the number of levels below the current Gaussian Pyramid level
        """
        self.levels_below = levels

    def set_levels_above(self, levels):
        r"""Informs the metric how many pyramid levels are above the current one

        Informs this metric the number of pyramid levels above the current one.
        The metric may change its behavior (e.g. number of inner iterations)
        accordingly

        Parameters
        ----------
        levels : int
            the number of levels above the current Gaussian Pyramid level
        """
        self.levels_above = levels

    def set_static_image(self, static_image, static_affine, static_spacing,
                         static_direction):
        r"""Sets the static image being compared against the moving one.

        Sets the static image. The default behavior (of this abstract class) is
        simply to assign the reference to an attribute, but
        generalizations of the metric may need to perform other operations

        Parameters
        ----------
        static_image : array, shape (R, C) or (S, R, C)
            the static image
        """
        self.static_image = static_image
        self.static_affine = static_affine
        self.static_spacing = cp.asarray(static_spacing)
        self.static_direction = static_direction

    def use_static_image_dynamics(self, original_static_image, transformation):
        r"""This is called by the optimizer just after setting the static image.

        This method allows the metric to compute any useful
        information from knowing how the current static image was generated
        (as the transformation of an original static image). This method is
        called by the optimizer just after it sets the static image.
        Transformation will be an instance of DiffeomorficMap or None
        if the original_static_image equals self.moving_image.

        Parameters
        ----------
        original_static_image : array, shape (R, C) or (S, R, C)
            original image from which the current static image was generated
        transformation : DiffeomorphicMap object
            the transformation that was applied to original image to generate
            the current static image
        """
        pass

    def set_moving_image(self, moving_image, moving_affine, moving_spacing,
                         moving_direction):
        r"""Sets the moving image being compared against the static one.

        Sets the moving image. The default behavior (of this abstract class) is
        simply to assign the reference to an attribute, but
        generalizations of the metric may need to perform other operations

        Parameters
        ----------
        moving_image : array, shape (R, C) or (S, R, C)
            the moving image
        """
        self.moving_image = moving_image
        self.moving_affine = moving_affine
        self.moving_spacing = cp.asarray(moving_spacing)
        self.moving_direction = moving_direction

    def use_moving_image_dynamics(self, original_moving_image, transformation):
        r"""This is called by the optimizer just after setting the moving image

        This method allows the metric to compute any useful
        information from knowing how the current static image was generated
        (as the transformation of an original static image). This method is
        called by the optimizer just after it sets the static image.
        Transformation will be an instance of DiffeomorficMap or None if
        the original_moving_image equals self.moving_image.

        Parameters
        ----------
        original_moving_image : array, shape (R, C) or (S, R, C)
            original image from which the current moving image was generated
        transformation : DiffeomorphicMap object
            the transformation that was applied to the original image to generate
            the current moving image
        """
        pass

    @abc.abstractmethod
    def initialize_iteration(self):
        r"""Prepares the metric to compute one displacement field iteration.

        This method will be called before any compute_forward or
        compute_backward call, this allows the Metric to pre-compute any useful
        information for speeding up the update computations. This
        initialization was needed in ANTS because the updates are called once
        per voxel. In Python this is unpractical, though.
        """

    @abc.abstractmethod
    def free_iteration(self):
        r"""Releases the resources no longer needed by the metric

        This method is called by the RegistrationOptimizer after the required
        iterations have been computed (forward and / or backward) so that the
        SimilarityMetric can safely delete any data it computed as part of the
        initialization
        """

    @abc.abstractmethod
    def compute_forward(self):
        r"""Computes one step bringing the reference image towards the static.

        Computes the forward update field to register the moving image towards
        the static image in a gradient-based optimization algorithm
        """

    @abc.abstractmethod
    def compute_backward(self):
        r"""Computes one step bringing the static image towards the moving.

        Computes the backward update field to register the static image towards
        the moving image in a gradient-based optimization algorithm
        """

    @abc.abstractmethod
    def get_energy(self):
        r"""Numerical value assigned by this metric to the current image pair

        Must return the numeric value of the similarity between the given
        static and moving images
        """


class CCMetric(SimilarityMetric):

    def __init__(self, dim, sigma_diff=2.0, radius=4, coord_axis=-1):
        r"""Normalized Cross-Correlation Similarity metric.

        Parameters
        ----------
        dim : int (either 2 or 3)
            the dimension of the image domain
        sigma_diff : the standard deviation of the Gaussian smoothing kernel to
            be applied to the update field at each iteration
        radius : int
            the radius of the squared (cubic) neighborhood at each voxel to be
            considered to compute the cross correlation
        """
        super(CCMetric, self).__init__(dim)
        self.sigma_diff = sigma_diff
        self.radius = radius
        if coord_axis not in [0, -1]:
            raise ValueError("coord_axis must be 0 or -1")
        self.coord_axis = coord_axis
        self._connect_functions()

    def _connect_functions(self):
        r"""Assign the methods to be called according to the image dimension

        Assigns the appropriate functions to be called for precomputing the
        cross-correlation factors according to the dimension of the input
        images
        """
        self.precompute_factors = cc.precompute_cc_factors
        self.compute_forward_step = cc.compute_cc_forward_step
        self.compute_backward_step = cc.compute_cc_backward_step
        self.reorient_vector_field = vfu.reorient_vector_field

    def initialize_iteration(self):
        r"""Prepares the metric to compute one displacement field iteration.

        Pre-computes the cross-correlation factors for efficient computation
        of the gradient of the Cross Correlation w.r.t. the displacement field.
        It also pre-computes the image gradients in the physical space by
        re-orienting the gradients in the voxel space using the corresponding
        affine transformations.
        """

        def invalid_image_size(image):
            min_size = self.radius * 2 + 1
            return any([size < min_size for size in image.shape])

        msg = ("Each image dimension should be superior to 2 * radius + 1."
               "Decrease CCMetric radius or increase your image size")

        if invalid_image_size(self.static_image):
            raise ValueError("Static image size is too small. " + msg)
        if invalid_image_size(self.moving_image):
            raise ValueError("Moving image size is too small. " + msg)

        self.factors = self.precompute_factors(self.static_image,
                                               self.moving_image,
                                               self.radius)

        if self.coord_axis == -1:
            self.gradient_moving = cp.empty(
                shape=(self.moving_image.shape) + (self.dim,), dtype=floating
            )

            for i, grad in enumerate(gradient(self.moving_image)):
                self.gradient_moving[..., i] = grad
        else:
            self.gradient_moving = cp.empty(
                shape=(self.dim,) + (self.moving_image.shape), dtype=floating
            )

            for i, grad in enumerate(gradient(self.moving_image)):
                self.gradient_moving[i] = grad

        # Convert moving image's gradient field from voxel to physical space
        if self.moving_spacing is not None:
            if self.coord_axis == -1:
                self.gradient_moving /= self.moving_spacing
            else:
                temp = self.moving_spacing.reshape((-1,) + (1,) * self.dim)
                self.gradient_moving /= temp
        if self.moving_direction is not None:
            self.reorient_vector_field(self.gradient_moving,
                                       self.moving_direction,
                                       coord_axis=self.coord_axis)

        if self.coord_axis == -1:
            self.gradient_static = cp.empty(
                shape=(self.static_image.shape) + (self.dim,), dtype=floating
            )
            for i, grad in enumerate(gradient(self.static_image)):
                self.gradient_static[..., i] = grad
        else:
            self.gradient_static = cp.empty(
                shape=(self.dim,) + (self.static_image.shape), dtype=floating
            )
            for i, grad in enumerate(gradient(self.static_image)):
                self.gradient_static[i] = grad

        # Convert moving image's gradient field from voxel to physical space
        if self.static_spacing is not None:
            if self.coord_axis == -1:
                self.gradient_static /= self.static_spacing
            else:
                temp = self.moving_spacing.reshape((-1,) + (1,) * self.dim)
                self.gradient_static /= temp

        if self.static_direction is not None:
            self.reorient_vector_field(self.gradient_static,
                                       self.static_direction,
                                       coord_axis=self.coord_axis)

    def free_iteration(self):
        r"""Frees the resources allocated during initialization
        """
        del self.factors
        del self.gradient_moving
        del self.gradient_static

    def compute_forward(self):
        """Computes one step bringing the moving image towards the static.

        Computes the update displacement field to be used for registration of
        the moving image towards the static image
        """
        displacement, self.energy = self.compute_forward_step(
            self.gradient_static,
            self.factors,
            self.radius,
            coord_axis=self.coord_axis,
        )
        if self.coord_axis == -1:
            for i in range(self.dim):
                displacement[..., i] = ndimage.filters.gaussian_filter(
                    displacement[..., i], self.sigma_diff
                )
        else:
            for i in range(self.dim):
                displacement[i] = ndimage.filters.gaussian_filter(
                    displacement[i], self.sigma_diff
                )
        return displacement

    def compute_backward(self):
        """Computes one step bringing the static image towards the moving.

        Computes the update displacement field to be used for registration of
        the static image towards the moving image
        """
        displacement, energy = self.compute_backward_step(
            self.gradient_moving,
            self.factors,
            self.radius,
            coord_axis=self.coord_axis,
        )
        if self.coord_axis == -1:
            for i in range(self.dim):
                displacement[..., i] = ndimage.filters.gaussian_filter(
                    displacement[..., i], self.sigma_diff
                )
        else:
            for i in range(self.dim):
                displacement[i] = ndimage.filters.gaussian_filter(
                    displacement[i], self.sigma_diff
                )
        return displacement

    def get_energy(self):
        r"""Numerical value assigned by this metric to the current image pair

        Returns the Cross Correlation (data term) energy computed at the
        largest iteration
        """
        return self.energy

# TODO:
# unimplemented classes:  EMMetric, SSDMetric
# unimplemented functions: v_cycle_2d, v_cycle_3d
