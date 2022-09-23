# Copyright JSB 2022 (jean-sebastien.brouillon@epfl.ch)

from conf import np, rrms_error
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
import matplotlib.colors as m_colors
import tikzplotlib
from examples.simple_case import define_simple_case, realistic_guess_simple
from examples.supply_demand import define_hard_case, realistic_guess_hard
from examples.watering import define_watering_case, realistic_guess_water
from examples.tensor import define_tensor_case, realistic_guess_tensor

"""
#
#   Tests for maximum likelihood estimation of Bayesian network inference
#
"""


def plot_results(_data, filename, logy=True):
    x, y = _data

    fig, ax = plt.subplots()
    plt.xscale("log")
    if logy:
        plt.yscale("log")

    ax.plot(x, y[0, :], color=list(m_colors.TABLEAU_COLORS)[0], alpha=1.0)
    ax.fill_between(x, y[1, :], y[2, :], color=list(m_colors.TABLEAU_COLORS)[0], alpha=.2)

    plt.savefig(filename, format="pdf")
    tikzplotlib.save(filename.replace(".pdf", ".tex"))
    plt.show()
    plt.clf()


def test_inference_simple(sample_size=10, uncertainty_factor=1.0):
    """
    Tests the inference of a continuous BN using a toy example.
    The function infers the supply of a good assuming the model in examples.simple_case

    :param sample_size: number of values for each node
    :param uncertainty_factor: reduce the uncertainty to simulate with lower noise
    """
    bn, exact_data = define_simple_case(sample_size, uncertainty_factor)

    # make a reasonable guess
    guess = realistic_guess_simple(bn)
    likelihoods = bn.infer(guess)

    return bn.data, exact_data, likelihoods, 0


def test_inference_hard(sample_size=10, tax_dimension=1, uncertainty_factor=1.0):
    """
    Tests the inference of a continuous BN using the example in examples.supply_demand.

    :param sample_size: number of values for each node
    :param tax_dimension: number of values for taxes and second dimension of price
    :param uncertainty_factor: reduce the uncertainty to simulate with lower noise
    """
    bn, exact_data, exact_tax = define_hard_case(sample_size, tax_dimension, uncertainty_factor)

    # make a reasonable guess
    guess = realistic_guess_hard(bn)
    likelihood_list = bn.infer(guess, verbose=True)

    return bn.data, exact_data, exact_tax, likelihood_list, 0


def test_inference(simple_case=test_inference_simple, hard_case=test_inference_hard, uncertainty_factor=0.001):
    """
    Display the results of tests of the inference of a continuous BN.

    :param simple_case: simple test function
    :param hard_case: hard test function
    :param uncertainty_factor: reduce the uncertainty to simulate with lower noise
    """
    # Simple case with very low noise
    _bn_data, _exact_data, _likelihoods, _ = simple_case(5, uncertainty_factor)
    print(np.mean(np.abs(_bn_data['supply'].squeeze()-_exact_data['supply'])) /
          np.mean(np.abs(_exact_data['supply'])))
    assert(np.allclose(np.mean(np.abs(_bn_data['supply'].squeeze()-_exact_data['supply'])) /
                       np.mean(np.abs(_exact_data['supply'])), 0, atol=0.02))

    # Hard case with very low noise
    _bn_data, _exact_data, _exact_tax, _likelihood_list, _ = hard_case(10, 1, uncertainty_factor)
    print(np.mean(np.abs(_bn_data['price'].squeeze() - np.array(_exact_data['price'].to_list()).squeeze())))
    print(_bn_data['tax'].squeeze(), _exact_tax.squeeze())
    assert(np.allclose(_bn_data['price'].squeeze(), np.array(_exact_data['price'].to_list()).squeeze(), atol=0.1))
    assert(np.allclose(_bn_data['tax'].squeeze(), _exact_tax.squeeze(), atol=0.5))


def test_watering(sample_size=None, uncertainty_factor=None):
    """
    Tests the inference of a continuous BN using the example in examples.watering.

    :param sample_size: number of values for each node
    :param uncertainty_factor: reduce the uncertainty to simulate with lower noise
    """
    seeds = range(34, 64)
    sample_size = [10] if sample_size is None else sample_size
    uncertainty_factor = [0.001] if uncertainty_factor is None else uncertainty_factor

    average_errors, average_times = [], []
    for i in tqdm(range(len(sample_size))):
        for j in range(len(uncertainty_factor)):
            errors, times = [], []
            for seed in seeds:
                np.random.seed(seed)
                bn, exact_data = define_watering_case(sample_size[i], uncertainty_factor[j])

                guess = realistic_guess_water(bn)
                bn.maximum_iterations = 10
                bn.stability_constant = 1e-4
                starting_time = time()
                bn.infer(guess, verbose=False)

                times.append(time() - starting_time)
                errors.append(np.mean(rrms_error(bn.data['irrad'].squeeze(), np.array(exact_data['irrad']).squeeze()))
                              + np.mean(np.abs(bn.data['rain'].squeeze() - np.array(exact_data['rain']).squeeze()))
                              / np.mean(np.abs(np.array(exact_data['rain']).squeeze())+1e-5))

            average_times.append([np.mean(times), np.min(times), np.max(times)])
            average_errors.append([np.mean(errors), np.min(errors), np.max(errors)])

    return np.transpose(average_times), np.transpose(average_errors)


def test_tensor(sample_size=10, uncertainty_factor=0.001):
    """
    Tests the inference of a continuous BN using the example in examples.tensor.

    :param sample_size: number of values for each node
    :param uncertainty_factor: reduce the uncertainty to simulate with lower noise
    """
    seeds = range(34, 64)
    n_its = range(2, 30)

    average_errors, average_times = [], []
    for i in tqdm(range(len(n_its))):
        errors, times = [], []
        for seed in seeds:
            np.random.seed(seed)
            bn, exact_data = define_tensor_case(sample_size, 2, uncertainty_factor)

            guess = realistic_guess_tensor(bn)
            bn.maximum_iterations = n_its[i]
            bn.tolerance = 0
            starting_time = time()
            bn.infer(guess, verbose=False)

            times.append(time() - starting_time)
            errors.append(np.mean(rrms_error(bn.data['va'].squeeze(), np.array(exact_data['va']).squeeze()))
                          + np.mean(rrms_error(bn.data['vb'].squeeze(), np.array(exact_data['vb']).squeeze())))

        average_times.append([np.mean(times), np.min(times), np.max(times)])
        average_errors.append([np.mean(errors), np.min(errors), np.max(errors)])

    return np.array(average_times)[:, 0], np.transpose(average_errors)


if __name__ == '__main__':
    np.set_printoptions(precision=2, floatmode='fixed')
    # test_inference(test_inference_simple, test_inference_hard, 0.001)

    # sizes = [2, 4, 10, 16, 25, 50, 80]
    # data = test_watering(sizes, [0.000001])[0]
    # data[1, :], data[2, :] = data[0, :], data[0, :]  # do not plot interval
    # plot_results((sizes, data), '../results/water_speed.pdf', False)

    uncertainties = [2e-13, 5e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
    plot_results((uncertainties, test_watering([10], uncertainties)[1]), '../results/water_robust.pdf', True)

    # plot_results(test_tensor(2000, 0.001), '../results/tensor_speed.pdf')
