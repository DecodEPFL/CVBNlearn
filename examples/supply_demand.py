# Copyright JSB 2022 (jean-sebastien.brouillon@epfl.ch)

import pandas as pd
from conf import np

"""
#
# This file defines a complex toy example to test the inference of a continuous BN.
# The example is based on the following simple price/supply/demand model.
# 
# supply  --,->  price  ->  demand
# tax     -'
# 
# Without randomness in human behavior, we have
#     price = (10 + (100 - supply)/10) * (1 + tax/100)
#     demand = 200 - 10*price
# 
# Behavioral randomness adds a Gaussian uncertainty to supply and demand.
# In this example, the price and tax are unknown.
# The inference of the tax has a Gaussian prior with a large uncertainty to make it almost non-informative.
#
"""

from src.bayesian_net import CVBNet


def define_hard_case(sample_size=10, tax_dimension=1, uncertainty_factor=1.0):
    """
    Defines the network, exact data and measured noisy data for the toy example.

    Gaussian uncertainty to supply and demand with standard deviation = 1% of mean (SNR = 10%).
    The inference of supply also benefit from a Gaussian prior with standard deviation of 10 around 100.

    A Gaussian prior is given to the tax with mean 0.1 and standard deviation 0.05
    :param sample_size: number of values for each node
    :param tax_dimension: number of values for taxes and second dimension of price
    :param uncertainty_factor: Gaussian uncertainty to supply and demand have
    standard deviation = uncertainty_factor*0.1% and uncertainty_factor*1% of mean, respectively.
    """
    # Data definition
    exact_data = pd.DataFrame().astype(object)
    exact_tax = np.random.normal(10, 3, size=(tax_dimension, 1))
    exact_data['supply'] = np.random.normal(100, 10, size=(sample_size, tax_dimension)).tolist()
    exact_data['price'] = ((10 + (100 - np.array(exact_data['supply'].to_list())) / 10)
                           @ (1 + exact_tax/100)/tax_dimension).tolist()
    exact_data['demand'] = (200 - 10*np.array(exact_data['price'].to_list())).tolist()

    data = pd.DataFrame()
    data['price'] = (pd.NA * np.array(exact_data['price'].to_list())).tolist()
    data['supply'] = np.random.normal(np.array(exact_data['supply'].to_list()), 0.1*uncertainty_factor).tolist()
    data['demand'] = np.random.normal(np.array(exact_data['demand'].to_list()), 1.0*uncertainty_factor).tolist()
    parameters = pd.DataFrame().astype(object)
    parameters['tax'] = [None]
    parameters['tax'].iloc[0] = [[None]] * tax_dimension

    # network creation
    bn = CVBNet()
    bn.from_pandas(data, parameters)

    # Distributions distribution
    bn.set_node('supply', distribution=("((x ** 2)/200) ** 0.2", "supply - 100"))
    bn.set_node('price', distribution=("((x ** 2)/0.02)",
                                       f"((10 + (100 - supply)/10) @ (1 + tax/100)/{tax_dimension}) - price"))
    bn.set_node('demand', distribution=("sqrt((x ** 2)/2)", "200 - 10*price - demand"))
    bn.set_node('tax', distribution=("(dot(x.T,x)/180000)", "10 - tax"))  # almost non-informative prior

    return bn, exact_data, exact_tax


def optimistic_guess_hard(bn, exact_data, exact_tax, uncertainty_factor=1.0):
    """
    Make an optimistic guess of exact_data, with its uncertainty
    This guess is centered around the exact value, and with std: std of the data * uncertainty_factor

    :param bn: The Bayesian network to infer starting from this guess
    :param exact_data: The exact data to be guessed
    :param exact_tax: The exact tax to be guessed
    :param uncertainty_factor: Gaussian uncertainty to price and demand have
    :return: data of initial guess
    """
    guess = bn.data.copy()
    guess['price'] = np.random.normal(np.array(exact_data['price'].to_list()), 0.1*uncertainty_factor)
    guess['tax'] = np.array(np.random.normal(exact_tax, 1*uncertainty_factor))

    return guess


def realistic_guess_hard(bn, uncertainty_factor=1.0):
    """
    Make a realistic guess of exact_data, with its uncertainty
    This guess is centered around the prior belief value, and with real std * uncertainty_factor

    :param bn: The Bayesian network to infer starting from this guess
    :param uncertainty_factor: Gaussian uncertainty to price and demand have
    :return: data of initial guess
    """
    guess = bn.data.copy()
    guess['price'] = np.random.normal(np.zeros_like(guess['price']) + 10, 1)
    guess['tax'] = np.random.normal(np.zeros_like(guess['tax']) + 10, 3)

    return guess


def benchmark_guess_hard(bn, exact_data, exact_tax, uncertainty_factor=1.0):
    """
    Wrapper function for all defined initial guesses

    :param bn: The Bayesian network to infer starting from this guess
    :param exact_data: The exact data to be guessed
    :param exact_tax: The exact tax to be guessed
    :param uncertainty_factor: Gaussian uncertainty to price and demand have
    :return: data of initial guess
    """
    return optimistic_guess_hard(bn, exact_data, exact_tax, 2.0)
    #return realistic_guess_hard(bn, 2.0)
