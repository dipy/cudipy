import numpy as np

from cudipy.segment.mrf import (
    ConstantObservationModel,
    IteratedConditionalModes,
)
from cudipy.sims.voxel import add_noise
from cudipy._utils import get_array_module

"""
The following two functions account for the majority of the computation time
    icm.icm_ising
    icm.prob_neighborhood
"""

class TissueClassifierHMRF(object):
    r"""
    This class contains the methods for tissue classification using the Markov
    Random Fields modeling approach
    """

    def __init__(self, save_history=False, verbose=True):

        self.save_history = save_history
        self.segmentations = []
        self.pves = []
        self.energies = []
        self.energies_sum = []
        self.verbose = verbose

    def classify(self, image, nclasses, beta, tolerance=None, max_iter=None):
        r"""
        This method uses the Maximum a posteriori - Markov Random Field
        approach for segmentation by using the Iterative Conditional Modes and
        Expectation Maximization to estimate the parameters.

        Parameters
        ----------
        image : ndarray,
                3D structural image.
        nclasses : int,
                number of desired classes.
        beta : float,
                smoothing parameter, the higher this number the smoother the
                output will be.
        tolerance: float,
                value that defines the percentage of change tolerated to
                prevent the ICM loop to stop. Default is 1e-05.
        max_iter : float,
                fixed number of desired iterations. Default is 100.
                If the user only specifies this parameter, the tolerance
                value will not be considered. If none of these two
                parameters

        Returns
        -------
        initial_segmentation : ndarray,
                3D segmented image with all tissue types
                specified in nclasses.
        final_segmentation : ndarray,
                3D final refined segmentation containing all
                tissue types.
        PVE : ndarray,
                3D probability map of each tissue type.
        """

        xp = get_array_module(image)
        nclasses = nclasses + 1  # One extra class for the background
        energy_sum = [1e-05]

        com = ConstantObservationModel()
        icm = IteratedConditionalModes()

        if not image.dtype.kind == 'f':
            image = image.astype(np.promote_types(image.dtype, np.float32))

        if image.max() > 1:
            # image = xp.interp(image, [0, image.max()], [0.0, 1.0])
            image = image - image.min()
            image /= image.max()

        mu, sigmasq = com.initialize_param_uniform(image, nclasses)
        p = xp.argsort(mu)
        mu = mu[p]
        sigmasq = sigmasq[p]

        neglogl = com.negloglikelihood(image, mu, sigmasq, nclasses)

        seg = icm.initialize_maximum_likelihood(neglogl)

        mu, sigmasq = com.seg_stats(image, seg, nclasses)

        zero = xp.zeros_like(image) + 0.001
        zero_noise = add_noise(zero, 10000, 1, noise_type="gaussian")
        image_gauss = xp.where(image == 0, zero_noise, image)

        final_segmentation = xp.empty_like(image)
        initial_segmentation = seg

        allow_break = max_iter is None or tolerance is not None

        if max_iter is None:
            max_iter = 100

        if tolerance is None:
            tolerance = 1e-05

        for i in range(max_iter):

            if self.verbose:
                print(">> Iteration: " + str(i))

            PLN = icm.prob_neighborhood(seg, beta, nclasses)
            PVE = com.prob_image(image_gauss, nclasses, mu, sigmasq, PLN)

            mu_upd, sigmasq_upd = com.update_param(image_gauss,
                                                   PVE, mu, nclasses)
            ind = xp.argsort(mu_upd)
            mu_upd = mu_upd[ind]
            sigmasq_upd = sigmasq_upd[ind]

            negll = com.negloglikelihood(image_gauss,
                                         mu_upd, sigmasq_upd, nclasses)
            final_segmentation, energy = icm.icm_ising(negll, beta, seg)

            if allow_break:
                energy_sum.append(float(energy[energy > -xp.inf].sum()))

            if self.save_history:
                self.segmentations.append(final_segmentation)
                self.pves.append(PVE)
                self.energies.append(energy)
                if allow_break:
                    self.energies_sum.append(energy_sum[-1])
                else:
                    self.energies_sum.append(float(energy[energy > -xp.inf].sum()))

            if allow_break and i > 5:

                e_sum = np.asarray(energy_sum)
                tol = tolerance * (np.amax(e_sum) - np.amin(e_sum))

                e_end = e_sum[e_sum.size - 5:]
                test_dist = np.abs(np.amax(e_end) - np.amin(e_end))

                if test_dist < tol:
                    break

            seg = final_segmentation
            mu = mu_upd
            sigmasq = sigmasq_upd

        PVE = PVE[1:, ...]
        PVE = xp.moveaxis(PVE, 0, -1)

        return initial_segmentation, final_segmentation, PVE
