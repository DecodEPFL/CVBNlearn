# Copyright JSB 2022 (jean-sebastien.brouillon@epfl.ch)

from conf import np

"""
#
# This file defines an example to test the inference of a continuous BN.
# The example is based on the following simple plant watering model.
# 
# exp(1atm-pressure)               -,
# sin(day of the academic year) -,--'-> mm of rain                     --,-> cm3 of water given
#                                '-> sun irradiance  -,-> soil humidity -'
#                                          FIR model -'
# 
# Without randomness in biological behavior, we have
# Rain ~ (1 - sin^2(day*pi/365)) * exp(1atm-pressure)	+ Laplace (outliers)
# Irradiance ~ sin(day*pi/365) + 1 		                + Laplace (outliers)
# humidity = base + K * irradiance			            + Gaussian
# Water given = need - rain + safety                    + Gaussian
#
# The data is
# exp(1atm-pressure)                                    : log-normal
# sin(day of the academic year)                         : exact
# sin(day of the academic year)                         : exact
# 
# Behavioral randomness adds uncertainty to water need.
# In this example, the rain and sun irradiance are unknown and will be inferred.
#
"""

from src.bayesian_net import CVBNet


def define_watering_case(sample_size=10, uncertainty_factor=1.0):
    """
    Defines the network, exact data and measured noisy data for the watering example.

    Uncertainty with standard deviation = 1% of mean is added to all variables

    :param sample_size: number of values for each node
    :param uncertainty_factor: Gaussian uncertainty to supply and demand have
    standard deviation = uncertainty_factor*0.1% and uncertainty_factor*1% of mean, respectively.
    """
    # Data definition
    exact_data = dict()
    exact_data['press'] = np.exp(1 - np.random.normal(np.ones(sample_size), 0.01))[:, None]
    exact_data['sun'] = (np.sin(np.linspace(0, np.pi, sample_size))**2)[:, None]
    exact_data['irrad'] = exact_data['sun'] + 1 - np.abs(np.random.laplace(np.zeros_like(exact_data['sun']), 0.2))
    exact_data['rain'] = np.maximum(np.random.laplace(3 * (1 - exact_data['sun']) * exact_data['press'], 3), 0)
    exact_data['model'] = np.tril(0.9 ** (np.arange(sample_size)[:, np.newaxis] - np.arange(sample_size)).clip(min=0))
    exact_data['humid'] = 10 + (exact_data['model'] @ exact_data['irrad'])/2
    exact_data['given'] = exact_data['humid'] - exact_data['rain'] + 2

    # network creation
    bn = CVBNet()

    # Distributions distribution
    bn.set_node('press', np.exp(np.random.normal(np.log(exact_data['press']), 0.0001*uncertainty_factor)),
                ("(x ** 2)/0.02", "log(press)"))
    bn.set_node('sun', exact_data['sun'], None)  # non-informative prior
    bn.set_node('irrad', np.nan * np.empty(sample_size), ("abs(x)*(1/0.2 + 100) + 100*x", "irrad - sun - 1"))
    bn.set_node('rain', np.nan * np.empty(sample_size),
                ("abs(x)/3", "(rain - 3 * press * (1 - sun))*(rain >= 0) + (100*rain)*(rain < 0)"))
    bn.set_node('model', exact_data['model'], None)  # non-informative prior
    bn.set_node('humid', np.random.normal(exact_data['humid'], 0.1*uncertainty_factor),
                (f"(x ** 2)/{0.02 * uncertainty_factor**2}", "humid - 10 - model @ irrad / 2"))
    bn.set_node('given', np.random.normal(exact_data['given'], 0.1*uncertainty_factor),
                (f"(x ** 2)/{0.02 * uncertainty_factor**2}", "given - (humid - rain + 2)"))

    return bn, exact_data


def optimistic_guess_water(bn, exact_data, uncertainty_factor=1.0):
    """
    Make an optimistic guess of exact_data, with its uncertainty
    This guess is centered around the exact value, and with std: std of the data * uncertainty_factor

    :param bn: The Bayesian network to infer starting from this guess
    :param exact_data: The exact data to be guessed
    :param uncertainty_factor: Gaussian uncertainty to price and demand have
    :return: data of initial guess
    """
    guess = bn.data.copy()
    guess['irrad'] = np.random.laplace(exact_data['irrad'], 2.0*uncertainty_factor)
    guess['rain'] = np.random.laplace(exact_data['rain'], 1.0*uncertainty_factor)

    return guess


def realistic_guess_water(bn, uncertainty_factor=1.0):
    """
    Make a realistic guess of exact_data, with its uncertainty
    This guess is centered around the prior belief value, and with real std * uncertainty_factor

    :param bn: The Bayesian network to infer starting from this guess
    :param uncertainty_factor: Gaussian uncertainty to price and demand have
    :return: data of initial guess
    """
    guess = bn.data.copy()

    sun = (np.sin(np.linspace(0, np.pi, bn.N))**2 + 1)[:, None]
    guess['irrad'] = sun - np.abs(np.random.laplace(np.zeros_like(sun), 0.2))
    guess['rain'] = np.random.laplace(3 * (2 - sun) * bn.data['press'], 3)

    return guess


def benchmark_guess_water(bn, exact_data, exact_tax, uncertainty_factor=1.0):
    """
    Wrapper function for all defined initial guesses

    :param bn: The Bayesian network to infer starting from this guess
    :param exact_data: The exact data to be guessed
    :param exact_tax: The exact tax to be guessed
    :param uncertainty_factor: Gaussian uncertainty to price and demand have
    :return: data of initial guess
    """
    #return optimistic_guess_hard(bn, exact_data, 1.0)
    return realistic_guess_hard(bn, 1.0)
