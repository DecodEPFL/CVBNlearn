# Copyright JSB 2022 (jean-sebastien.brouillon@epfl.ch)

from conf import np, rrms_error
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as m_colors
import tikzplotlib
from benchmarks.tests import test_bn
from benchmarks.tests import test_discretize
from benchmarks.tests import test_sampling
from benchmarks.tests import test_zo_gradient

"""
#
#   Compare CVBNlean with sampling, GLD and discretization
#
"""


def plot_results(data, filename, var="price", logy=False):
    colors = dict()
    for col in range(len(data.keys())):
        colors[list(data.keys())[col]] = list(m_colors.TABLEAU_COLORS)[col]

    fig, ax = plt.subplots()
    plt.xscale("log")
    if logy:
        plt.yscale("log")

    for k, v in data.items():
        ax.plot(v["time"], v[var][0], color=colors[k], alpha=1.0)
    ax.legend(list(data.keys()))

    for k, v in data.items():
        ax.fill_between(v["time"], v[var][1], v[var][2], color=colors[k], alpha=.2)

    plt.savefig(filename, format="pdf")
    tikzplotlib.save(filename.replace(".pdf", ".tex"))
    plt.show()
    plt.clf()


def compare_accuracy_fixed_speed(tests=None, n_its=None, uncertainty_factor=1e-10):
    """
    Compares the accuracy of several methods for perfect knowledge, 2 unknown variables (of dimensions 2 and 1),
    and when the time allocated to the algorithm is fixed.

    The average accuracy for several run times are returned for all unknowns, as well as their min/max.

    :param tests: list of 4 test functions to be ran, hard case by default
    :param n_its: base number of iterations to execute (proportional to run time)
    :param uncertainty_factor: reduce the uncertainty to simulate with lower noise
    """
    seeds = [31, 113, 131]
    n_its = [0, 100, 200, 300, 400, 500, 600, 800,
             1000, 2000, 5000, 10000, 20000, 50000, 100000] if n_its is None else n_its
    n_its_factors = [1/100, 1, 1, 1/5]

    if tests is None:
        tests = [test_bn.test_inference_hard, test_discretize.test_inference_hard,
                 test_sampling.test_inference_hard, test_zo_gradient.test_inference_hard]
    names = ["cvbn", "discrete", "sampled", "zo grad"]

    average_errors_price = ([], [], [], [])
    average_errors_tax = ([], [], [], [])
    average_time = ([], [], [], [])

    for n_it in tqdm(n_its):
        for i in range(len(tests)):
            errors_price, errors_tax, times = [], [], []
            for seed in seeds:
                np.random.seed(seed)
                bn_data, exact_data, exact_tax, likelihood_list, elapsed_time = \
                    tests[i](2, 1, uncertainty_factor, int(n_it*n_its_factors[i]))

                times.append(elapsed_time)
                errors_price.append(np.mean(rrms_error(bn_data['price'].squeeze(),
                                                       np.array(exact_data['price'].to_list()).squeeze())))
                errors_tax.append(rrms_error(bn_data['tax'].squeeze(), exact_tax.squeeze()))

            average_errors_price[i].append([np.mean(errors_price), np.min(errors_price), np.max(errors_price)])
            average_errors_tax[i].append([np.mean(errors_tax), np.min(errors_tax), np.max(errors_tax)])
            average_time[i].append(np.mean(times))

    return_data = dict()
    for i in range(len(tests)):
        return_data[names[i]] = {"price": np.array(average_errors_price[i]).T,
                                 "tax": np.array(average_errors_tax[i]).T,
                                 "time": np.array(average_time[i])}
    return return_data


def compare_scaling(tests=None, uncertainty_factor=1e-10):
    """
    Compares the scaling to higher dimensions of several methods for perfect knowledge,
    when the final accuracy allocated to the algorithm is fixed.

    The average run time for several dimension numbers, as well as their min/max are returned.

    :param tests: list of test functions to be ran, hard case by default
    :param uncertainty_factor: reduce the uncertainty to simulate with lower noise
    """
    seeds = [31, 113, 131]
    n_its = [[2, 1000], [4, 2000], [10, 5000], [16, 10000], [25, 20000], [50, 50000], [80, 100000]]
    n_its_factors = [1/100, 1, 500, 100]

    if tests is None:
        tests = [test_bn.test_inference_hard, test_zo_gradient.test_inference_hard,
                 test_discretize.test_inference_hard, test_sampling.test_inference_hard]
    names = ["cvbn", "zo grad", "discrete", "sampled"]

    average_errors_price = ([], [], [], [])
    average_errors_tax = ([], [], [], [])
    average_time = ([], [], [], [])

    for n_it in tqdm(n_its):
        for i in range(len(tests)):
            errors_price, errors_tax, times = [], [], []
            for seed in seeds:
                np.random.seed(seed)
                bn_data, exact_data, exact_tax, likelihood_list, elapsed_time = \
                    tests[i](n_it[0], 1, uncertainty_factor, int(n_it[1]*n_its_factors[i]))

                times.append(elapsed_time)
                errors_price.append(np.mean(rrms_error(bn_data['price'].squeeze(),
                                                          np.array(exact_data['price'].to_list()).squeeze())))
                errors_tax.append(rrms_error(bn_data['tax'].squeeze(), exact_tax.squeeze()))

            average_errors_price[i].append([np.mean(errors_price), np.min(errors_price), np.max(errors_price)])
            average_errors_tax[i].append([np.mean(errors_tax), np.min(errors_tax), np.max(errors_tax)])
            average_time[i].append(np.mean(times))


    return_data = dict()
    for i in range(len(tests)):
        return_data[names[i]] = {"price": np.tile(average_time[i], (3, 1)),
                                 "tax": np.tile(average_time[i], (3, 1)),
                                 "time": np.array(n_its)[:, 0]}

    # To check that errors stay somewhat constant
    # return_data = dict()
    # for i in range(len(tests)):
    #     return_data[names[i]] = {"price": np.array(average_errors_price[i]).T,
    #                              "tax": np.array(average_errors_tax[i]).T,
    #                              "time": np.array(average_time[i])}
    return return_data


def compare_robustness(tests=None):
    """
    Compares the robustness to uncertainty of several methods for noisy data,
    when the algorithms reach convergence (t->∞).

    The average accuracy time for several uncertainty levels are returned, as well as their min/max.

    :param tests: list of test functions to be ran, hard case by default
    """
    seeds = [31, 113, 131]
    u_factors = [1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
    n_its = [50, 20000, 2000, 2000]

    if tests is None:
        tests = [test_bn.test_inference_hard, test_zo_gradient.test_inference_hard]
    names = ["cvbn", "zo grad"]

    average_errors_price, average_errors_tax, average_time = ([], [], []), ([], [], []), ([], [], [])
    for uncertainty in tqdm(u_factors):
        for i in range(len(tests)):
            errors_price, errors_tax, times = [], [], []
            for seed in seeds:
                np.random.seed(seed)
                bn_data, exact_data, exact_tax, likelihood_list, elapsed_time = \
                    tests[i](10, 1, uncertainty, n_its[i])

                times.append(elapsed_time)
                errors_price.append(np.mean(rrms_error(bn_data['price'].squeeze(),
                                                          np.array(exact_data['price'].to_list()).squeeze())))
                errors_tax.append(rrms_error(bn_data['tax'].squeeze(), exact_tax.squeeze()))

            average_errors_price[i].append([np.mean(errors_price), np.min(errors_price), np.max(errors_price)])
            average_errors_tax[i].append([np.mean(errors_tax), np.min(errors_tax), np.max(errors_tax)])
            average_time[i].append(np.mean(times))

    return_data = dict()
    for i in range(len(tests)):
        return_data[names[i]] = {"price": np.array(average_errors_price[i]).T,
                                 "tax": np.array(average_errors_tax[i]).T,
                                 "time": np.array(u_factors)}
    return return_data



if __name__ == '__main__':

    _data = compare_accuracy_fixed_speed()
    print(_data)
    plot_results(_data, "../../results/compare_hard_global_price.pdf", "price")
    plot_results(_data, "../../results/compare_hard_global_tax.pdf", "tax")
    del _data['cvbn']
    plot_results(_data, "../../results/compare_hard_others_price.pdf", "price")
    plot_results(_data, "../../results/compare_hard_others_tax.pdf", "tax")

    # This can be very long with disctretize and sampling
    _data = compare_scaling(tests=[test_bn.test_inference_hard, test_zo_gradient.test_inference_hard])
    print(_data)
    plot_results(_data, '../../results/compare_hard_scaling.pdf', "price", False)

    _data = compare_robustness()
    print(_data)
    plot_results(_data, '../../results/compare_hard_robust_price.pdf', "price", True)
    plot_results(_data, '../../results/compare_hard_robust_tax.pdf', "tax", True)
