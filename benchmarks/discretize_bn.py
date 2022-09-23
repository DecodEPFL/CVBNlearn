# Copyright JSB 2022 (jean-sebastien.brouillon@epfl.ch)

from tqdm import tqdm
from conf import np
from src.bayesian_net import CVBNet


class CVBNetDiscretized(CVBNet):
    """
    #
    #   Class for representing discretized continuous variables bayesian networks
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
    
        # determine grid size to remain below maximum_iterations
        n_variables = np.sum([self.data[node].size for node in nodes_to_infer])
        grid_size = int(np.floor(np.power(self.maximum_iterations, 1.0/n_variables)))
    
        # Generate meshgrid
        grid_points = dict()
        for node in nodes_to_infer:
            grid_points[node] = np.linspace(initial_guess[node] - guess_deviation[node],
                                            initial_guess[node] + guess_deviation[node], grid_size+1)
            grid_points[node] = (grid_points[node][:-1] + grid_points[node][1:]) / 2
    
        # Loop of maximum_iterations iterations, stopping when tolerance condition is met
        pbar = tqdm(total=self.maximum_iterations) if verbose else None
        likelihoods_list = [-np.inf]
        max_likelihood = -np.inf
        current_data = dict()
        for node in self.data.keys():
            current_data[node] = self.data[node].copy()
    
        for i in range(grid_size ** n_variables):
            if verbose:
                pbar.update(1)
    
            variables_passed = 0
            for node in nodes_to_infer:
                for k in range(self.data[node].size):
                    kn = variables_passed + k
                    current_data[node][k] = \
                        grid_points[node][int(np.floor(i / (grid_size ** kn)) % grid_size), k]
    
                variables_passed = variables_passed + self.data[node].size
    
            # Generate sample likelihood
            likelihoods_list.append(self._log_likelihood(current_data))
            if likelihoods_list[-1] > max_likelihood:
                max_likelihood = likelihoods_list[-1]
                for node in self.data.keys():
                    self.data[node] = current_data[node].copy()
    
        del likelihoods_list[0]
    
        # return likelihood progression
        if verbose:
            pbar.close()
        return likelihoods_list
