# Copyright JSB 2022 (jean-sebastien.brouillon@epfl.ch)

from conf import np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as m_colors
import tikzplotlib
from examples.supply_demand import define_hard_case, realistic_guess_hard

"""
#
#   Tests for variance estimation of Bayesian network inference
#
"""


def plot_results(data, filename, logy=False):
    x, var_exact, var_estimated, var_fast = data
    fig, ax = plt.subplots()
    plt.xscale("log")
    if logy:
        plt.yscale("log")

    ax.plot(x, var_exact[0, :], color=list(m_colors.TABLEAU_COLORS)[0], alpha=1.0)
    ax.plot(x, var_estimated[0, :], color=list(m_colors.TABLEAU_COLORS)[1], alpha=1.0)
    ax.plot(x, var_fast[0, :], color=list(m_colors.TABLEAU_COLORS)[2], alpha=1.0)

    ax.legend(["exact", "estimated", "fast"])

    ax.fill_between(x, var_exact[1, :], var_exact[2, :],
                    color=list(m_colors.TABLEAU_COLORS)[0], alpha=.2)
    ax.fill_between(x, var_estimated[1, :], var_estimated[2, :],
                    color=list(m_colors.TABLEAU_COLORS)[1], alpha=.2)
    ax.fill_between(x, var_fast[1, :], var_fast[2, :],
                    color=list(m_colors.TABLEAU_COLORS)[2], alpha=.2)

    plt.savefig(filename, format="pdf")
    tikzplotlib.save(filename.replace(".pdf", ".tex"))
    plt.show()
    plt.clf()


def test_variance(uncertainty_factors=None, variance_to_compute=None):
    """
    Tests the variance estimation of estimates in a continuous BN using a toy example.
    The function infers the tax on a good assuming the same model as "test_inference_hard".
    The variance is computed using resampling, total variance and fast total variance methods,
    for a list of noise levels and for given latent variables.

    A Gaussian prior is given to the tax with mean 0.1 and standard deviation 0.05

    :param uncertainty_factors: list of factors reducing the uncertainty to simulate with lower noise,
    [0.001] is default
    :param variance_to_compute: list of names of variables for which to compute the variance
    """
    seeds = np.arange(34, 64)
    sample_size = 100
    n_tries = 10
    tax_dimension = 2
    uncertainty_factors = [0.001] if uncertainty_factors is None else uncertainty_factors
    variance_to_compute = ["tax", "price"] if variance_to_compute is None else variance_to_compute

    variances_sample, variances_calc, variances_fast = [], [], []
    for u_factor in tqdm(uncertainty_factors):
        var_sample, var_est, var_fast = [], [], []
        for seed in seeds:
            np.random.seed(seed)

            # Data definition
            bn, exact_data, exact_tax = define_hard_case(sample_size, tax_dimension, u_factor)

            variances, results = dict(), dict()
            for name in variance_to_compute:
                variances[name], variances[name + "_fast"], results[name] = [], [], []

            for i in range(n_tries):
                # New dataset
                bn.set_node('supply', np.random.normal(np.array(exact_data['supply'].to_list()), 0.1*u_factor),
                            ("(((x ** 2)/200) ** 0.2)", "supply - 100"))
                bn.set_node('price', np.nan * np.array(exact_data['price'].to_list()),
                            (f"((x ** 2)/0.02/{u_factor**2})",
                             f"((10 + (100 - supply)/10) @ (1 + tax/100)/{tax_dimension}) - price"))
                bn.set_node('demand', np.random.normal(np.array(exact_data['demand'].to_list()), 1.0*u_factor),
                            (f"sqrt((x ** 2)/2/{u_factor**2})", "200 - 10*price - demand"))
                bn.set_node('tax', np.array([[np.nan]] * tax_dimension),
                            distribution=("(dot(x.T,x)/180000)", "10 - tax"))  # almost non-informative prior

                # Compute MLE + variance
                bn.infer(realistic_guess_hard(bn), verbose=False)
                bn.maximum_iterations = 50
                estimate_variance, _ = bn.compute_variance(variance_to_compute, 0.01*u_factor)
                fast_variance, _ = bn.compute_variance(variance_to_compute, 0.01*u_factor, fast=True)

                # add variances and expectations to the list
                for node in variance_to_compute:
                    variances[node].append(estimate_variance[node])
                    variances[node + "_fast"].append(fast_variance[node])
                    results[node].append(bn.data[node].copy())

            # Compute empirical variance over all tries
            variance_empirical, expectation_empirical = dict(), dict()
            for name in variance_to_compute:
                expectation_empirical[name] = np.zeros_like(results[name][0].flatten())
                variance_empirical[name] = np.outer(expectation_empirical[name], expectation_empirical[name])
                for i in range(n_tries):
                    expectation_empirical[name] = expectation_empirical[name] + results[name][i].flatten() / n_tries
                    variance_empirical[name] = variance_empirical[name] + np.outer(results[name][i].flatten(),
                                                                                   results[name][i].flatten())
                variance_empirical[name] = (variance_empirical[name] / n_tries
                                            - np.outer(expectation_empirical[name], expectation_empirical[name]))

            var_sample.append(np.linalg.norm(variance_empirical["tax"]))
            var_est.append(np.linalg.norm(np.sum(variances["tax"], axis=0)))  # / n_tries
            var_fast.append(np.linalg.norm(np.sum(variances["tax_fast"], axis=0)))  # / n_tries

        variances_sample.append([np.mean(var_sample), np.min(var_sample), np.max(var_sample)])
        variances_calc.append([np.mean(var_est), np.min(var_est), np.max(var_est)])
        variances_fast.append([np.mean(var_fast), np.min(var_fast), np.max(var_fast)])

    return uncertainty_factors, np.array(variances_sample).T, np.array(variances_calc).T, np.array(variances_fast).T


if __name__ == '__main__':
    np.set_printoptions(precision=2, floatmode='fixed')

    plot_results(test_variance(np.logspace(-4.5, 0.5, num=12), ['tax']), '../results/variance.pdf', True)
