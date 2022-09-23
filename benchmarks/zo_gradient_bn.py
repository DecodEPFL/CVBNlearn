# Copyright JSB 2022 (jean-sebastien.brouillon@epfl.ch)

from tqdm import tqdm
from conf import np
from src.bayesian_net import CVBNet


class CVBNetZO(CVBNet):
    """
    #
    #   Class for representing continuous variables bayesian networks
    #   and performing inference with zero order gradient.
    #
    """
    def __init__(self, other=None):
        """
        Constructor for an empty Bayesian Network

        :param other: other net to copy
        """
        CVBNet.__init__(self)
        if other is None:
            self.data = dict()
            self.distributions = dict()
            self.shapes = dict()
            self.N = 0
            self.stability_constant = 1e-4
            # self.numerical_derivation_constant = 1e-2
        else:
            self.data = other.data.copy()
            self.distributions = other.distributions.copy()
            self.shapes = other.shapes.copy()
            self.N = other.N
            self.stability_constant = other.stability_constant
            # self.numerical_derivation_constant = other.numerical_derivation_constant

            self.tolerance = other.tolerance
            self.maximum_iterations = other.maximum_iterations

    def infer(self, initial_guess=None, guess_deviation=None, verbose=False):
        """
        Infers all the missing values in the data.
        Parameters of conditional distributions to estimate must be additional root nodes.
        If some distributions are not specified, an uninformative prior will be used.

        :param initial_guess: Initial guess for the unknown variables, as a dictionary or
        as a tuple of two DataFrames (1 for data 1 for parameters)
        :param guess_deviation: deviation of the values between the samples.
        :param verbose: Turns on the verbose.
        :return: List of the likelihoods at each iteration
        """
        nodes_to_infer = self._process_initial_guess(initial_guess)

        # Loop of maximum_iterations iterations, stopping when tolerance condition is met
        pbar = tqdm(total=self.maximum_iterations) if verbose else None
        likelihoods_list = [self._log_likelihood()]
        step_size = self.stability_constant
        while len(likelihoods_list) < self.maximum_iterations:
            if verbose:
                pbar.update(1)

            step_size = step_size*(1.0 - 1.0/self.maximum_iterations)
            data_perturbed = self.data.copy()

            for node in nodes_to_infer:
                perturbation = np.random.normal(0, guess_deviation[node])
                data_perturbed[node] = self.data[node].copy() + step_size * perturbation

                likelihood = self._log_likelihood(data_perturbed)
                data_perturbed[node] = self.data[node].copy() + perturbation \
                    * (likelihood - likelihoods_list[-1])

            likelihoods_list.append(self._log_likelihood(data_perturbed))
            self.data = data_perturbed

            # stopping criterion
            if len(likelihoods_list) > 1:
                if np.abs(likelihoods_list[-2] - likelihoods_list[-1])/np.abs(likelihoods_list[-1]) < self.tolerance:
                    break

        # return likelihood progression
        if verbose:
            pbar.close()
        return likelihoods_list
