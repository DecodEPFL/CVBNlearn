# Copyright JSB 2022 (jean-sebastien.brouillon@epfl.ch)

from conf import np, verbose
from time import time
from examples.simple_case import define_simple_case, benchmark_guess_simple
from examples.supply_demand import define_hard_case, benchmark_guess_hard
from tests.test_examples import test_inference

"""
#
#   Tests Tests for edition features of Bayesian network class
#
"""


def test_inference_simple(sample_size=10, uncertainty_factor=1.0, steps=50):
    """
    Tests the inference of a continuous BN using a toy example.
    The function infers the supply of a good assuming the model in examples.simple_case

    :param sample_size: number of values for each node
    :param uncertainty_factor: reduce the uncertainty to simulate with lower noise
    :param steps: number of iterations
    """
    bn, exact_data = define_simple_case(sample_size, uncertainty_factor)

    # make a reasonable guess
    guess = benchmark_guess_simple(bn, exact_data, uncertainty_factor)
    bn.maximum_iterations = steps
    bn.tolerance = 0
    start_time = time()
    likelihoods = bn.infer(guess, verbose=verbose)
    elapsed_time = time() - start_time

    return bn.data, exact_data, likelihoods, elapsed_time


def test_inference_hard(sample_size=10, tax_dimension=1, uncertainty_factor=1.0, steps=50):
    """
    Tests the inference of a continuous BN using the example in examples.hard_case.

    :param sample_size: number of values for each node
    :param tax_dimension: number of values for taxes and second dimension of price
    :param uncertainty_factor: reduce the uncertainty to simulate with lower noise
    :param steps: number of iterations
    """
    bn, exact_data, exact_tax = define_hard_case(sample_size, tax_dimension, uncertainty_factor)

    # make a reasonable guess
    guess = benchmark_guess_hard(bn, exact_data, exact_tax, uncertainty_factor)
    bn.maximum_iterations = steps
    bn.tolerance = 0
    start_time = time()
    likelihood_list = bn.infer(guess, verbose=verbose)
    elapsed_time = time() - start_time

    return bn.data, exact_data, exact_tax, likelihood_list, elapsed_time


if __name__ == '__main__':
    np.set_printoptions(precision=2, floatmode='fixed')
    test_inference(test_inference_simple, test_inference_hard, 0.001)
