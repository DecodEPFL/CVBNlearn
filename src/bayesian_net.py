# Copyright JSB 2022 (jean-sebastien.brouillon@epfl.ch)

import pandas as pd
from tqdm import tqdm
from conf import np, data_type
from src.constrained_functional import ConstrainedDistribution


class CVBNet:
    """
    #
    #   Class for representing continuous variables bayesian networks
    #
    """

    def __init__(self, tolerance=1e-6, maximum_iterations=50):
        """
        Constructor for an empty Bayesian Network

        :param tolerance: relative tolerance for the stopping criterion of the inference algorithm
        when the relative difference in likelihood between two iterations |Li - Li+1|/|Li|
        is greater than tolerance, the algorithm in the method "infer" stops.
        :param maximum_iterations: maximum number of iterations performed by the method "infer".
        """
        self.data = dict()
        self.distributions = dict()
        self.shapes = dict()
        self.N = 0
        self.stability_constant = 1e-3
        # self.numerical_derivation_constant = 1e-2

        self.tolerance = tolerance
        self.maximum_iterations = maximum_iterations

    def from_pandas(self, samples, parameters, distributions=None):
        """
        Load data from pandas dataframe.
        A node is created for each column, with its name.
        Columns can have an object data type to implement vector/matrix-valued variables.

        There are two types of data: samples and parameters.
        There are N samples for time-varying variables
        Time-constant variables must be in a dataframe of length 1, but can be matrix valued.
        To make a df matrix valued use df['column'] = [None], followed by df['column'].iloc[0] = matrix

        Unknown data to infer must be left as np.nan or pd.NA or None

        :param samples: dataframe of time-variant variables
        :param parameters: dataframe of time-invariant parameters
        :param distributions: (optional) dictionary containing all the distribution functionnals
        :return: self
        """
        self.N = len(samples.index)
        if not parameters.empty and len(parameters.index) != 1:
            raise ValueError("length of parameters DataFrame must be 1")

        samples = samples.astype(object)
        parameters = parameters.astype(object)

        # Replace NAs or nans with None
        # those values mean that there is a value to infer
        # Set an element in the data dictionary for each column
        for key in samples.columns.values.tolist():
            numpy_type_array = np.array(samples[key].to_list())
            # Need "==" operator with None here to obtain boolean array
            numpy_type_array = np.where(numpy_type_array == None, np.nan, numpy_type_array)
            numpy_type_array = np.where(pd.isna(numpy_type_array), np.nan, numpy_type_array)
            self.set_node(key, numpy_type_array.astype(data_type),
                          None if distributions is None else distributions[key])
        for key in parameters.columns.values.tolist():
            numpy_type_array = np.array(parameters[key].iloc[0])
            # Need "==" operator with None here to obtain boolean array
            numpy_type_array = np.where(numpy_type_array == None, np.nan, numpy_type_array)
            numpy_type_array = np.where(pd.isna(numpy_type_array), np.nan, numpy_type_array)
            self.set_node(key, numpy_type_array.astype(data_type),
                          None if distributions is None else distributions[key])

        return self

    def to_pandas(self):
        """
        Create dataframe based on network's data
        A column is created for each node, with its name.
        Columns have an object data type to implement vector/matrix-valued variables.
        There are two types of data: samples and parameters.
        Time-varying variables are in a dataframe of length N.
        Time-constant variables are in a dataframe of length 1.

        :return: samples is a dataframe of time-variant variables,
        parameters is a dataframe of time-invariant parameters
        """
        samples = pd.DataFrame(dtype=object)
        parameters = pd.DataFrame(dtype=object)

        # Set each column to an element of the data dictionary
        for k, v in self.data.items():
            if v.shape[0] == self.N:
                samples[k] = v.tolist()
            else:
                parameters[k] = v.tolist()

        # Replace Nones by NAs
        samples.where(pd.notnull(samples), pd.NA)
        parameters.where(pd.notnull(parameters), pd.NA)

        return samples, parameters

    def set_node(self, name, value=None, distribution=None, is_distribution_log=True):
        """
        Sets the data or distribution of a node "name".

        To keep the previous values or distribution, set the corresponding argument to None.
        The default distribution is an uninformative prior.

        :param name: name of the node to set
        :param value: value to give to the node's variable as a numpy array
        :param distribution: functional describing the distribution depending on other nodes
        :param is_distribution_log: Flag that is True if the self.distributions contains log(p) instead of p
        :return: self
        """
        if self.N == 0:
            self.N = value.shape[0]

        if value is not None:
            if len(value.shape) < 2:
                value = value[:, None]
            self.data[name] = value.astype(data_type)
        if distribution is not None:
            if type(distribution) is not ConstrainedDistribution:
                distribution = ConstrainedDistribution(distribution, is_distribution_log)
            self.distributions[name] = distribution
        elif name not in self.distributions.keys():
            self.distributions[name] = ConstrainedDistribution((f"dot(x.T,x)*{self.stability_constant}", name),
                                                               is_distribution_log)

        return self

    def get_node(self, name):
        """
        Gets the data and distribution of a name "name".
        Data is an array of size N or 1 in the first dimension.
        Size N corresponds to time-varying variables and 1 to invariant parameters

        :param name: name of the node to set
        :return: value to give to the node's variable, and
        functional describing the distribution depending on other nodes
        """
        returned_array, returned_distribution = None, None

        if name in self.data.keys():
            returned_array = self.data[name]
        if name in self.distributions.keys():
            returned_distribution = self.distributions[name]

        return returned_array, returned_distribution

    def infer(self, initial_guess=None, verbose=False):
        """
        Infers all the missing values in the data.
        Parameters of conditional distributions to estimate must be additional root nodes.
        If some distributions are not specified, an uninformative prior will be used.

        :param initial_guess: Initial guess for the unknown variables, as a dictionary or
        as a tuple of two DataFrames (1 for data 1 for parameters)
        :param verbose: Turns on the verbose.
        :return: List of the likelihoods at each iteration
        """
        nodes_to_infer = self._process_initial_guess(initial_guess)

        # Initialize likelihood from guess
        pbar = tqdm(total=self.maximum_iterations) if verbose else None
        likelihoods_list = [self._log_likelihood()]

        # Loop of maximum_iterations iterations, stopping when tolerance condition is met
        residual = -np.inf
        while np.abs((self._stop_criterion() - residual)/self._stop_criterion()) > self.tolerance \
                and len(likelihoods_list) < self.maximum_iterations:
            if verbose:
                pbar.update(1)

            # apply AILS
            for node in nodes_to_infer:
                scale_stack, bias_stack = np.empty((0, self.data[node].size)), np.empty(0)
                for distribution in self.distributions.values():
                    # Compute linearization
                    pre_scale, post_scale, bias = self._linearize(node, distribution)
                    scale, bias, _ = self._reweight(node, distribution, pre_scale, post_scale, bias)

                    scale_stack = np.vstack((scale_stack, scale))
                    bias_stack = np.concatenate((bias_stack, bias))

                estimate = np.linalg.lstsq(scale_stack, bias_stack, rcond=-1)[0]
                self.data[node] = estimate.reshape(self.data[node].shape, order='F')

            # save likelihood
            likelihoods_list.append(self._log_likelihood())

        return likelihoods_list


    def compute_variance(self, nodes=None, relative_std=0.01, fast=False, verbose=False):
        """
        Computes the variance of the estimate given by self.infer().
        WARNING: ONLY EXECUTE THIS FUNCTION WHEN BN.DATA IS CLOSE TO THE REAL DATA (e.g. after executing self.infer())
        Numerical instability is often very strong otherwise.

        :param nodes: list of the names of variables to compute the variance of.
        :param relative_std: Relative standard deviation for the resampling of the probabilities' linearization.
        :param fast: Use fast inverse approximation to speed up the method. ONLY USE WITH SMALL relative_std.
        :param verbose: Turns on the verbose.
        :return: dict with covariance matrices for each vectorized variable in nodes
        """
        if nodes is None or len(nodes)==0:
            return dict()

        # Initialize likelihood from guess
        pbar = tqdm(total=self.maximum_iterations) if verbose else None

        estimate_variance, estimate_expectation, likelihoods_lists = dict(), dict(), dict()
        for node in nodes:
            # Compute inverse of scale.T @ scale once
            scale_mode, bias_mode = np.empty((0, self.data[node].size)), np.empty(0)
            for distribution in self.distributions.values():
                pre_scale, post_scale, bias = self._linearize(node, distribution)
                scale, bias, _ = self._reweight(node, distribution, pre_scale, post_scale, bias)
                scale_mode = np.vstack((scale_mode, scale))
                bias_mode = np.concatenate((bias_mode, bias))
            scale_centered = scale_mode - np.mean(scale_mode, axis=0)[None, :]
            original_inverse = np.linalg.inv(scale_centered.T @ scale_centered)

            # Initialize matrices
            estimate_variance[node] = np.zeros((self.data[node].size, self.data[node].size))
            estimate_expectation[node] = np.zeros((self.data[node].size, 1))
            likelihoods_lists[node] = [self._log_likelihood()]

            # Do self.maximum_iterations resamplings
            for i in (tqdm(range(self.maximum_iterations)) if verbose else range(self.maximum_iterations)):
                scale_stack, bias_stack = np.empty((0, self.data[node].size)), np.empty(0)
                likelihood = 0
                for distribution in self.distributions.values():
                    # Compute linearization
                    pre_scale, post_scale, bias = self._linearize(node, distribution)
                    depends_on_node = not (np.sum(np.abs(pre_scale)) == 0 or np.sum(np.abs(post_scale)) == 0)

                    # Add random perturbations
                    post_scale += np.random.normal(0*post_scale, relative_std*np.abs(post_scale))
                    pre_scale += np.random.normal(0*pre_scale, relative_std*np.abs(pre_scale))
                    scale, bias, w = self._reweight(node, distribution, pre_scale, post_scale, bias)

                    # compute likelihood
                    probability = distribution.get_function()[0]
                    likelihood -= np.sum(w) if depends_on_node else (np.sum(distribution(self.data)) \
                        if distribution.is_log() else np.sum(-np.log(distribution(self.data))))

                    scale_stack = np.vstack((scale_stack, scale))
                    bias_stack = np.concatenate((bias_stack, bias))

                likelihoods_lists[node].append(likelihood)

                residuals = scale_stack @ self.data[node] - bias_stack[:, None]

                # Centering data
                scale_centered = scale_stack - np.mean(scale_stack, axis=0)[None, :]
                inverse_matrix = original_inverse - (scale_stack - scale_mode).T @ (scale_stack - scale_mode) if fast \
                    else np.linalg.inv(scale_centered.T @ scale_centered)

                expected_bias = - inverse_matrix @ scale_centered.T @ residuals
                estimate_variance[node] += (inverse_matrix * (np.std(residuals)**2) + expected_bias.T @ expected_bias) \
                    * np.exp(likelihood - likelihoods_lists[node][0])
                estimate_expectation[node] += expected_bias * np.exp(likelihood - likelihoods_lists[node][0])

        # Expectation over variations in scale
        for node in nodes:
            probability_normalizer = (np.sum(np.exp(np.array(likelihoods_lists[node][1:])
                                                    - likelihoods_lists[node][0])))
            estimate_expectation[node] = estimate_expectation[node] / probability_normalizer
            estimate_variance[node] = estimate_variance[node] / probability_normalizer
            estimate_variance[node] = estimate_variance[node] - estimate_expectation[node].T @estimate_expectation[node]

        return estimate_variance, likelihoods_lists

    def _process_initial_guess(self, initial_guess):
        """
        Sets the data of the bn according to an initial guess

        :param initial_guess: Initial guess for the unknown variables, as a dictionary or
        as a tuple of two DataFrames (1 for data 1 for parameters)
        :return: nodes with data missing that need to be inferred
        """
        # Check which nodes are missing data
        nodes_to_infer = list()
        for node, value in self.data.items():
            if np.any(np.isnan(value)):
                nodes_to_infer.append(node)

        # Process initial value
        samples, parameters = pd.DataFrame(), pd.DataFrame()
        if type(initial_guess) != dict and initial_guess is not None:
            for i in range(2):
                if len(initial_guess[i].index) == 1:
                    parameters = initial_guess[i]
                elif len(initial_guess[i].index) == self.N:
                    samples = initial_guess[i]
                else:
                    raise ValueError("length of DataFrames must be 1 or net.N")
            self.from_pandas(samples, parameters)
        elif type(initial_guess) == dict:
            for node in nodes_to_infer:
                self.data[node] = initial_guess[node]
        else:
            for node in nodes_to_infer:
                self.data[node] = np.where(self.data[node] == np.nan, self.data[node], 0)

        return nodes_to_infer

    def _log_likelihood(self, data_used=None):
        """
        Evaluates the joint likelihood for a specific data realization

        :param data_used: data to evaluate the likelihood at (default is self.data)
        :return: log-likelihood
        """
        data_used = self.data if data_used is None else data_used

        likelihood = 0
        for distribution in self.distributions.values():
            likelihood = likelihood - np.sum(distribution(data_used)) if distribution.is_log() \
                else np.sum(-np.log(distribution(data_used)))
        return likelihood

    def _stop_criterion(self, data_used=None):
        """
        Evaluates the norm of the residuals for a specific data realization

        :param data_used: data to evaluate the likelihood at (default is self.data)
        :return: log-likelihood
        """
        data_used = self.data if data_used is None else data_used

        criterion = 0
        for distribution in self.distributions.values():
            criterion = criterion - np.sum(distribution.residual(data_used))
        return criterion

    def _linearize(self, node, distribution):
        """
        Find the scales and bias matrices S1, S2, and B such that S1 @ x['node'] @ S2 + B
        is equal to the inner function of distributions[node]

        :param node: name of the node giving x['node']
        :param distribution: distribution function to use
        :return: tuple of scales and bias matrices
        """
        data_const = self.data.copy()
        data_loop = self.data.copy()
        _f = distribution.get_function()[1]

        # Extract constant w.r.t.   with x['node'] = 0
        data_const[node] = data_const[node] * 0 if len(self.data[node].shape) >= 2 else data_const[node][:, None] * 0
        bias = _f(data_const)

        # Compute the columns of the scale by changing each element of x['node'] separately
        data_loop[node] = data_const[node] * 0
        pre_scale = np.empty((bias.shape[0], data_loop[node].shape[0]))
        post_scale = np.empty((data_loop[node].shape[1], 1 if len(bias.shape) < 2 else bias.shape[1]))

        # Obtain the multiplier by setting one element of x['node'] to 1
        for i in range(pre_scale.shape[1]):
            data_loop[node][i, 0] = 1.0
            pre_scale[:, i] = (_f(data_loop) - bias).flatten('F')[:pre_scale.shape[0]]
            data_loop[node][i, 0] = 0.0

        for i in range(post_scale.shape[0]):
            data_loop[node][0, i] = 1.0
            post_scale[i, :] = (_f(data_loop) - bias).flatten('F')[::pre_scale.shape[0]]
            data_loop[node][0, i] = 0.0

        # Normalize preferably the first matrix if it is tall
        if pre_scale.shape[0] > pre_scale.shape[1] and pre_scale[0, 0] != 0:
            pre_scale = pre_scale/pre_scale[0, 0]
        elif pre_scale[0, 0] != 0:
            post_scale = post_scale/pre_scale[0, 0]
        if len(bias.shape) < 2:
            bias = bias[:, None]

        return pre_scale, post_scale, ((np.sum(pre_scale, axis=1) != 0)[:, None] * bias)

    def _reweight(self, node, distribution, pre_scale, post_scale, bias):
        """
        Reweight the scale and bias matrices S and B such that (S @ x['node'] + B).T @ (S @ x['node'] + B)
        is equal to the function of distributions[node]

        :param node: node that is linearized for
        :param distribution: distribution to use for weighting
        :param pre_scale: pre-multiplication factor from self._linearize()
        :param post_scale: post-multiplication factor from self._linearize()
        :param bias: additive bias from self._linearize()
        :return: reweighted scale and bias
        """
        scale, bias = np.kron(post_scale.T, pre_scale), bias.flatten('F')

        w = distribution(self.data) if distribution.is_log() else -np.log(distribution(self.data))
        w_mat = (w.flatten('F') + self.stability_constant) / \
            ((scale @ self.data[node].flatten('F') + bias) ** 2 + self.stability_constant)

        return np.sqrt(w_mat)[:, None] * scale, - np.sqrt(w_mat) * bias, w
