# Copyright JSB 2022 (jean-sebastien.brouillon@epfl.ch)

import pandas as pd
from conf import np

"""
#
# This file defines a toy example to test the inference of a continuous BN.
# The example is based on the following simple price/supply/demand model.
# 
# supply  --,->  price  ->  demand
# tax     -'
# 
# Without randomness in human behavior, we have
#     price = (10 + (100 - supply)/10) * (1 + tax/100)
#     demand = 200 - 10*price
# 
# Behavioral randomness adds a Gaussian uncertainty to price and demand. In this example, the supply is unknown.
# The inference of supply also benefit from a Gaussian prior with standard deviation of 10 around 100.
#
"""

from src.bayesian_net import CVBNet


def define_simple_case(sample_size=10, uncertainty_factor=1.0):
    """
    Defines the network, exact data and measured noisy data for the toy example.

    :param sample_size: number of values for each node
    :param uncertainty_factor: Gaussian uncertainty to price and demand have
    standard deviation = uncertainty_factor*0.1% and uncertainty_factor*1% of mean, respectively.
    """
    # Data definition
    exact_data = pd.DataFrame()
    exact_data['supply'] = np.random.normal(100, 10, size=sample_size)
    exact_data['tax'] = 20*np.random.choice(2, sample_size)
    exact_data['price'] = (10 + (100 - exact_data['supply']) / 10) * (1 + exact_data['tax']/100)
    exact_data['demand'] = 200 - 10*exact_data['price']

    data = pd.DataFrame()
    data['supply'] = pd.NA * exact_data['price']
    data['tax'] = exact_data['tax']
    data['price'] = np.random.normal(exact_data['price'], 0.1*uncertainty_factor)
    data['demand'] = np.random.normal(exact_data['demand'], 1.0*uncertainty_factor)

    # network creation
    bn = CVBNet()
    bn.from_pandas(data, pd.DataFrame(dtype=object))

    # Distributions distribution
    bn.set_node('supply', distribution=("dot(x.T,x)/200", "supply - 100"))
    bn.set_node('price', distribution=("dot(x.T,x)/0.02", "((10 + (100 - supply)/10) * (1 + tax/100)) - price"))
    bn.set_node('demand', distribution=("dot(x.T,x)/2", "200 - 10*price - demand"))

    return bn, exact_data


def optimistic_guess_simple(bn, exact_data, uncertainty_factor=1.0):
    """
    Make an optimistic guess of exact_data, with its uncertainty
    This guess is centered around the exact value, and with std: std of the data * uncertainty_factor

    :param bn: The Bayesian network to infer starting from this guess
    :param exact_data: The exact data to be guessed
    :param uncertainty_factor: Gaussian uncertainty to price and demand have
    :return: data of initial guess
    """
    guess = bn.data.copy()
    guess['supply'] = np.random.normal(np.array(exact_data['supply'].to_list()), 10*uncertainty_factor)[:, None]

    return guess


def realistic_guess_simple(bn, uncertainty_factor=1.0):
    """
    Make a realistic guess of exact_data, with its uncertainty
    This guess is centered around the prior belief value, and with real std * uncertainty_factor

    :param bn: The Bayesian network to infer starting from this guess
    :param uncertainty_factor: Gaussian uncertainty to price and demand have
    :return: data of initial guess
    """
    # make a reasonable guess, with its uncertainty
    guess = bn.data.copy()
    guess['supply'] = np.random.normal(np.zeros_like(guess['supply']) + 100, 10*uncertainty_factor)

    return guess


def benchmark_guess_simple(bn, exact_data, uncertainty_factor=1.0):
    """
    Wrapper function for all defined initial guesses

    :param bn: The Bayesian network to infer starting from this guess
    :param exact_data: The exact data to be guessed
    :param uncertainty_factor: Gaussian uncertainty to price and demand have
    :return: data of initial guess
    """
    return realistic_guess_simple(bn, uncertainty_factor)
