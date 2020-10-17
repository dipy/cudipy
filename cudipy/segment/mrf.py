import math

import numpy as np

import cupy as cp
from cudipy.segment._icm_kernels import (
    _get_icm_prob_class_kernel,
    _get_icm_weights,
)
from cudipy._utils import get_array_module


class ConstantObservationModel(object):
    r"""
    Observation model assuming that the intensity of each class is constant.
    The model parameters are the means $\mu_{k}$ and variances $\sigma_{k}$
    associated with each tissue class. According to this model, the observed
    intensity at voxel $x$ is given by $I(x) = \mu_{k} + \eta_{k}$ where $k$
    is the tissue class of voxel $x$, and $\eta_{k}$ is a Gaussian random
    variable with zero mean and variance $\sigma_{k}^{2}$. The observation
    model is responsible for computing the negative log-likelihood of
    observing any given intensity $z$ at each voxel $x$ assuming the voxel
    belongs to each class $k$. It also provides a default parameter
    initialization.
    """

    def __init__(self):
        r""" Initializes an instance of the ConstantObservationModel class
        """
        pass

    def initialize_param_uniform(self, image, nclasses):
        r""" Initializes the means and variances uniformly

        The means are initialized uniformly along the dynamic range of
        `image`. The variances are set to 1 for all classes

        Parameters
        ----------
        image : array
            3D structural image
        nclasses : int
            number of desired classes

        Returns
        -------
        mu : array
            1 x nclasses, mean for each class
        var : array
            1 x nclasses, variance for each class.
            Set to 1.0 for all classes.
        """
        xp = get_array_module(image)
        float_dtype = xp.promote_types(image.dtype, np.float32)
        mu = xp.arange(0, 1, 1 / nclasses, dtype=float_dtype) * image.ptp()
        sigma = xp.ones((nclasses,), dtype=float_dtype)

        return mu, sigma

    def seg_stats(self, input_image, seg_image, nclass):
        r""" Mean and standard variation for N desired  tissue classes

        Parameters
        ----------
        input_image : ndarray
            3D structural image
        seg_image : ndarray
            3D segmented image
        nclass : int
            number of classes (3 in most cases)

        Returns
        -------
        mu, var : ndarrays
            1 x nclasses dimension
            Mean and variance for each class

        """
        xp = get_array_module(input_image)
        mu = xp.zeros(nclass, dtype=input_image.dtype)
        std = xp.zeros(nclass, dtype=input_image.dtype)
        for i in range(nclass):
            v = input_image[seg_image == i]
            if v.size > 0:
                mu[i] = v.mean()
                std[i] = v.var()
        return mu, std

    def negloglikelihood(self, image, mu, sigmasq, nclasses):
        r""" Computes the gaussian negative log-likelihood of each class at
        each voxel of `image` assuming a gaussian distribution with means and
        variances given by `mu` and `sigmasq`, respectively (constant models
        along the full volume). The negative log-likelihood will be written
        in `nloglike`.

        Parameters
        ----------
        image : ndarray
            3D gray scale structural image
        mu : ndarray
            mean of each class
        sigmasq : ndarray
            variance of each class
        nclasses : int
            number of classes

        Returns
        -------
        nloglike : ndarray
            4D negloglikelihood for each class in each volume
        """
        xp = get_array_module(image)
        float_dtype = xp.promote_types(image.dtype, np.float32)
        nloglike = xp.zeros((nclasses,) + image.shape, dtype=float_dtype)

        for l in range(nclasses):
            nloglike[l, ...] = _negloglikelihood(image, mu[l], sigmasq[l])

        return nloglike

    def prob_image(self, img, nclasses, mu, sigmasq, P_L_N):
        r""" Conditional probability of the label given the image

        Parameters
        -----------
        img : ndarray
            3D structural gray-scale image
        nclasses : int
            number of tissue classes
        mu : ndarray
            1 x nclasses, current estimate of the mean of each tissue class
        sigmasq : ndarray
            1 x nclasses, current estimate of the variance of each
            tissue class
        P_L_N : ndarray
            4D probability map of the label given the neighborhood.

        Previously computed by function prob_neighborhood

        Returns
        --------
        P_L_Y : ndarray
            4D probability of the label given the input image
        """
        xp = get_array_module(img)
        P_L_Y = xp.zeros_like(P_L_N)
        P_L_Y_norm = xp.zeros_like(img)
        out = xp.zeros_like(img)

        for l in range(nclasses):
            P_L_Y[l] = _prob_image(img, mu[l], sigmasq[l], P_L_N[l])
            P_L_Y_norm += P_L_Y[l]

        # TODO: why is this guard needed here, but not in the CPU case?
        P_L_Y_norm[P_L_Y_norm == 0] = 1  # avoid divide by 0

        P_L_Y /= P_L_Y_norm[np.newaxis, ...]

        return P_L_Y

    def update_param(self, image, P_L_Y, mu, nclasses):
        r""" Updates the means and the variances in each iteration for all
        the labels. This is for equations 25 and 26 of Zhang et. al.,
        IEEE Trans. Med. Imag, Vol. 20, No. 1, Jan 2001.

        Parameters
        -----------
        image : ndarray
            3D structural gray-scale image
        P_L_Y : ndarray
            4D probability map of the label given the input image
            computed by the expectation maximization (EM) algorithm
        mu : ndarray
            1 x nclasses, current estimate of the mean of each tissue
            class.
        nclasses : int
            number of tissue classes

        Returns
        --------
        mu_upd : ndarray
                1 x nclasses, updated mean of each tissue class
        var_upd : ndarray
                1 x nclasses, updated variance of each tissue class
        """
        xp = get_array_module(image)
        float_dtype = xp.promote_types(image.dtype, np.float32)

        # lower memory if we don't broadcast to larger arrays
        mu_upd = xp.zeros(nclasses, dtype=float_dtype)
        var_upd = xp.zeros(nclasses, dtype=float_dtype)

        # fuse simple computations to reduce kernel launch overhead
        @cp.fuse()
        def _helper(image, mu, p_l_y):
            var_num = image - mu
            var_num *= var_num
            var_num *= p_l_y
            return var_num

        for l in range(nclasses):
            mu_num = P_L_Y[l] * image
            var_num = _helper(image, mu[l], P_L_Y[l])

            denom = xp.sum(P_L_Y[l])
            mu_upd[l] = xp.sum(mu_num) / denom
            var_upd[l] = xp.sum(var_num) / denom

        return mu_upd, var_upd


# TODO: use np.finfo(imaged.dtype).eps instead of hardcoded values below?
def _negloglikelihood(image, mu, sigmasq):
    xp = get_array_module(image)
    eps = 1e-8  # We assume images normalized to 0-1
    eps_sq = 1e-16  # Maximum precision for double.
    float_dtype = np.promote_types(image.dtype, np.float32)
    var = sigmasq
    fvar = float(var)  # host scalar
    c = math.log(math.sqrt(2.0 * np.pi * fvar))

    if fvar < eps_sq:
        neglog = xp.empty(image.shape, dtype=float_dtype)
        small_mask = xp.abs(image - mu) < eps
        neglog[small_mask] = 1 + c
        neglog[~small_mask] = np.inf
    else:

        # fuse simple computations to reduce kernel launch overhead
        @cp.fuse()
        def _helper(image, mu, var, c):
            neglog = image - mu
            neglog *= neglog
            neglog /= 2.0 * var
            neglog += c
            return neglog

        neglog = _helper(image, mu, var, c)

    return neglog


_gaussian_kern = cp.ElementwiseKernel(
    in_params="W image, W mu, W var, P prob",
    out_params="W out",
    operation="""
    out = image - mu;
    out *= out;
    out /= 2.0 * var;
    out = exp(-out);
    out /= sqrt(2.0 * M_PI * var);
    out *= prob;
    """,
    name="cudipy_gaussian_kernel",
)


# TODO: use np.finfo(imaged.dtype).eps instead of hardcoded values below?
def _prob_image(image, mu, sigmasq, P_L_N, out=None):
    xp = get_array_module(image)
    eps = 1e-8  # We assume images normalized to 0-1
    eps_sq = 1e-16  # Maximum precision for double.
    float_dtype = np.promote_types(image.dtype, np.float32)
    var = sigmasq
    fvar = float(var)  # host scalar version of var
    if out is not None:
        if out.shape != image.shape:
            raise ValueError("out must have the same shape as image")
        if out.dtype != image.dtype:
            raise ValueError("out must have the same dtype as image")
    if fvar < eps_sq:
        if out is None:
            gaussian = xp.zeros(image.shape, dtype=float_dtype)
        else:
            gaussian = out
            gaussian[...] = 0
        small_mask = xp.abs(image - mu) < eps
        gaussian[small_mask] = P_L_N[small_mask]
    else:
        if out is None:
            gaussian = xp.empty_like(image)
        else:
            gaussian = out
        _gaussian_kern(image, mu, var, P_L_N, gaussian)
    return gaussian


class IteratedConditionalModes(object):
    def __init__(self):
        pass

    def initialize_maximum_likelihood(self, nloglike):
        r""" Initializes the segmentation of an image with given
            neg-loglikelihood

        Initializes the segmentation of an image with neglog-likelihood field
        given by `nloglike`. The class of each voxel is selected as the one
        with the minimum neglog-likelihood (i.e. maximum-likelihood
        segmentation).

        Parameters
        ----------
        nloglike : ndarray
            4D shape, nloglike[x, y, z, k] is the likelihhood of class k
            for voxel (x, y, z)

        Returns
        --------
        seg : ndarray
            3D initial segmentation
        """
        return nloglike.argmin(axis=0).astype(np.int16)

    def icm_ising(self, nloglike, beta, seg):
        r""" Executes one iteration of the ICM algorithm for MRF MAP
        estimation. The prior distribution of the MRF is a Gibbs
        distribution with the Potts/Ising model with parameter `beta`:

        https://en.wikipedia.org/wiki/Potts_model

        Parameters
        ----------
        nloglike : ndarray
            4D shape, nloglike[x, y, z, k] is the negative log likelihood
            of class k at voxel (x, y, z)
        beta : float
            positive scalar, it is the parameter of the Potts/Ising
            model. Determines the smoothness of the output segmentation.
        seg : ndarray
            3D initial segmentation. This segmentation will change by one
            iteration of the ICM algorithm

        Returns
        -------
        new_seg : ndarray
            3D final segmentation
        energy : ndarray
            3D final energy
        """
        xp = get_array_module(nloglike)

        nclasses = nloglike.shape[0]

        float_dtype = xp.promote_types(
            nloglike.dtype, np.float32
        )  # Note: use float64 as in Dipy?
        energies = xp.empty(nloglike.shape, dtype=float_dtype)

        p_l_n = xp.empty(seg.shape, dtype=float_dtype)

        icm_weights = _get_icm_weights(seg.ndim, beta, float_dtype)
        int_type = "size_t" if seg.size > 1 << 31 else "int"
        prob_kernel = _get_icm_prob_class_kernel(icm_weights.shape, int_type)

        for classid in range(nclasses):
            prob_kernel(seg, icm_weights, classid, p_l_n)
            energies[classid, ...] = p_l_n + nloglike[classid, ...]

        # The code below is equivalent, but more efficient than:
        #     return energies.argmin(-1), energies.min(-1)

        # reshape to energies to (voxels, classes)
        vol_shape = energies.shape[1:]
        energies = energies.reshape(energies.shape[0], -1)
        argm = energies.argmin(0)
        # get min using the argmin indices (via advanced indexing)
        energy = energies[argm, cp.arange(energies.shape[1])]
        # restore original spatial domain shape
        new_seg = argm.reshape(vol_shape)
        energy = energy.reshape(vol_shape)
        return new_seg, energy

    def prob_neighborhood(self, seg, beta, nclasses, float_dtype=np.float32):
        r""" Conditional probability of the label given the neighborhood
        Equation 2.18 of the Stan Z. Li book (Stan Z. Li, Markov Random Field
        Modeling in Image Analysis, 3rd ed., Advances in Pattern Recognition
        Series, Springer Verlag 2009.)

        Parameters
        -----------
        seg : ndarray
            3D tissue segmentation derived from the ICM model
        beta : float
            scalar that determines the importance of the neighborhood and
            the spatial smoothness of the segmentation.
            Usually between 0 to 0.5
        nclasses : int
            number of tissue classes

        Returns
        --------
        PLN : ndarray
            4D probability map of the label given the neighborhood of the
            voxel.
        """
        xp = get_array_module(seg)
        p_l_n = xp.empty(seg.shape, dtype=float_dtype)

        icm_weights = _get_icm_weights(seg.ndim, beta, float_dtype)
        int_type = "size_t" if seg.size > 1 << 31 else "int"
        prob_kernel = _get_icm_prob_class_kernel(icm_weights.shape, int_type)

        PLN = xp.zeros((nclasses,) + seg.shape, dtype=float_dtype)

        for classid in range(nclasses):
            prob_kernel(seg, icm_weights, classid, p_l_n)
            xp.exp(-p_l_n, out=p_l_n)
            PLN[classid, ...] = p_l_n

        PLN /= PLN.sum(0, keepdims=True)

        return PLN
