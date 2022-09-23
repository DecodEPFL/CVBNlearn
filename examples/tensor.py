# Copyright JSB 2022 (jean-sebastien.brouillon@epfl.ch)

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


def define_tensor_case(sample_size=10, dimension=4, uncertainty_factor=1.0):
    """
    Defines the network, exact data and measured noisy data for the toy example.

    :param sample_size: number of values for each node
    :param dimension: size of the tensor
    :param uncertainty_factor: Gaussian uncertainty to price and demand have
    standard deviation = uncertainty_factor*0.1% and uncertainty_factor*1% of mean, respectively.
    """
    # Data definition
    exact_data = dict()
    exact_data['va'] = np.random.normal(1, 0.5, size=(dimension, 1))
    exact_data['vb'] = np.random.normal(1, 0.5, size=(1, dimension))
    exact_data['vb'] = exact_data['vb'] / np.sum(exact_data['vb'])
    exact_data['z'] = np.random.normal(np.zeros((sample_size, dimension)), 1.0)
    exact_data['y'] = exact_data['z'] @ np.outer(exact_data['va'], exact_data['vb'])

    # network creation
    bn = CVBNet()

    # Distributions distribution
    bn.set_node('z', exact_data['z'], (f"(x ** 2)/2000000", "z"))
    bn.set_node('va', np.nan * exact_data['va'], None)
    bn.set_node('vb', np.nan * exact_data['vb'], (f"abs(x)/(2 * {uncertainty_factor**2})","1 - sum(vb , 1)"))
    bn.set_node('y', np.random.normal(exact_data['y'], 0*uncertainty_factor),
                (f"(x ** 2)/(2 * {uncertainty_factor**2})", "y - z @ va @ vb"))

    return bn, exact_data



def realistic_guess_tensor(bn, uncertainty_factor=1.0):
    """
    Wrapper function for all defined initial guesses

    :param bn: The Bayesian network to infer starting from this guess
    :param exact_data: The exact data to be guessed
    :param uncertainty_factor: Gaussian uncertainty to price and demand have
    :return: data of initial guess
    """
    guess = bn.data.copy()
    guess["va"] = np.ones_like(bn.data["va"])
    guess["vb"] = np.ones_like(bn.data["vb"])

    return guess
