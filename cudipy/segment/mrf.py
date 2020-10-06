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
        sigma : array
            1 x nclasses, standard deviation for each class.
            Set up to 1.0 for all classes.
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
        mu, std : ndarrays
            1 x nclasses dimension
            Mean and standard deviation for each class

        """
        xp = get_array_module(input_image)
        mu = xp.zeros(nclass)
        std = xp.zeros(nclass)
        for i in range(nclass):
            v = input_image[seg_image == i]
            if v.size > 0:
                mu[i] = v.mean()
                std[i] = v.std()
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
        nloglike = xp.zeros(image.shape + (nclasses,), dtype=float_dtype)

        for l in range(nclasses):
            nloglike[..., l] = _negloglikelihood(image, mu[l], sigmasq[l])

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

        for l in range(nclasses):

            P_L_Y[..., l] = _prob_image(img, mu[l], sigmasq[l], P_L_N[..., l])
            P_L_Y_norm += P_L_Y[..., l]

        # TODO: why is this guard needed here, but not in the CPU case?
        P_L_Y_norm[P_L_Y_norm == 0] = 1  # avoid divide by 0

        P_L_Y /= P_L_Y_norm[..., np.newaxis]

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

        for l in range(nclasses):
            mu_num = P_L_Y[..., l] * image
            var_num = image - mu[l]
            var_num *= var_num
            var_num *= P_L_Y[..., l]

            denom = xp.sum(P_L_Y[..., l])
            mu_upd[l] = xp.sum(mu_num) / denom
            var_upd[l] = xp.sum(var_num) / denom

        return mu_upd, var_upd


# TODO: use np.finfo(imaged.dtype).eps instead of hardcoded values below?
# TODO: use ElementWise Kernel for better efficiency
def _negloglikelihood(image, mu, sigmasq):
    xp = get_array_module(image)
    eps = 1e-8  # We assume images normalized to 0-1
    eps_sq = 1e-16  # Maximum precision for double.
    float_dtype = np.promote_types(image.dtype, np.float32)
    var = float(sigmasq)
    if var < eps_sq:
        neglog = xp.empty(image.shape, dtype=float_dtype)
        small_mask = xp.abs(image - mu) < eps
        neglog[small_mask] = 1 + math.log(math.sqrt(2.0 * np.pi * var))
        neglog[~small_mask] = np.inf
    else:
        neglog = image - mu
        neglog *= neglog
        neglog /= 2 * var
        neglog += math.log(math.sqrt(2.0 * np.pi * var))
    return neglog


# TODO: use np.finfo(imaged.dtype).eps instead of hardcoded values below?
# TODO: use ElementWise Kernel for better efficiency
def _prob_image(image, mu, sigmasq, P_L_N):
    xp = get_array_module(image)
    eps = 1e-8  # We assume images normalized to 0-1
    eps_sq = 1e-16  # Maximum precision for double.
    float_dtype = np.promote_types(image.dtype, np.float32)
    var = float(sigmasq)
    if var < eps_sq:
        gaussian = xp.zeros(image.shape, dtype=float_dtype)
        small_mask = xp.abs(image - mu) < eps
        gaussian[small_mask] = 1
    else:
        gaussian = image - mu
        gaussian *= gaussian
        gaussian /= 2 * var
        gaussian = xp.exp(-gaussian)
        gaussian /= math.sqrt(2 * np.pi * var)

    return gaussian * P_L_N


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
        return nloglike.argmin(axis=-1).astype(np.int16)

    # TODO: create custom kernel similar to _icm_ising for better efficiency
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

        nclasses = nloglike.shape[-1]

        float_dtype = xp.promote_types(
            seg.dtype, np.float32
        )  # Note: use float64 as in Dipy?
        energies = xp.empty(nloglike.shape).astype(float_dtype)

        p_l_n = xp.empty(seg.shape, dtype=float_dtype)

        icm_weights = _get_icm_weights(seg.ndim, beta, float_dtype)
        int_type = "size_t" if seg.size > 1 << 31 else "int"
        prob_kernel = _get_icm_prob_class_kernel(icm_weights.shape, int_type)

        for classid in range(nclasses):
            prob_kernel(seg, icm_weights, classid, p_l_n)
            energies[..., classid] = p_l_n + nloglike[..., classid]

        new_seg = energies.argmin(-1)
        energy = energies.min(-1)
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

        PLN = xp.zeros(seg.shape + (nclasses,), dtype=float_dtype)

        for classid in range(nclasses):
            prob_kernel(seg, icm_weights, classid, p_l_n)
            xp.exp(-p_l_n, out=p_l_n)
            PLN[..., classid] = p_l_n

        PLN /= PLN.sum(-1, keepdims=True)

        return PLN
