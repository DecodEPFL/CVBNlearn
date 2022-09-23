# Copyright JSB 2022 (jean-sebastien.brouillon@epfl.ch)

from conf import np
from src.bayesian_net import CVBNet
from examples.simple_case import define_simple_case, realistic_guess_simple
from examples.supply_demand import define_hard_case, realistic_guess_hard

"""
#
#   Tests for maximum likelihood estimation of Bayesian network inference
#
"""


def test_linearize():
    """
    Tests the linearization w.r.t. one node of the network
    Compares the output of the inner multi-linear function to the one of the linearization
    when the values of all other nodes are constant.

    1d, 2d linear and tri-linear cases are tested.
    """
    # test function
    def test_results(t_dist, t_bn, t_pre_scale, t_xs, t_post_scale, t_bias):
        res = t_dist.get_function()[1](t_bn.data)
        np.testing.assert_allclose(res.flatten('F'),
                                   np.kron(t_post_scale.T, t_pre_scale) @ t_xs.flatten('F') + t_bias.flatten('F'))
        if len(res.shape) < 2:
            res = res[:, None]
        if len(t_xs.shape) < 2:
            t_xs = t_xs[:, None]
        np.testing.assert_allclose(res, t_pre_scale @ t_xs @ t_post_scale + t_bias)

    # network creation
    bn = CVBNet()

    # 1d, linear case
    bn.set_node('abc', np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                ("exp(-dot(x,x)/200)", "2*abc + 5"))
    xs, dist = bn.get_node('abc')

    pre_scale, post_scale, bias = bn._linearize('abc', dist)
    test_results(dist, bn, pre_scale, xs, post_scale, bias)

    # 2d, linear case
    bn.set_node('abc', np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [1, 2, 3, 4, 5]]),
                ("exp(-dot(x,x)/200)", "0.5*abc + 1"))
    xs, dist = bn.get_node('abc')

    pre_scale, post_scale, bias = bn._linearize('abc', dist)
    test_results(dist, bn, pre_scale, xs, post_scale, bias)

    # reset network
    bn = CVBNet()

    # 1d, tri-linear case
    bn.set_node('a', np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                ("exp(-dot(x,x)/2)", "2*b@a@c + 2 + c"))
    bn.set_node('b', np.array([[0.9, 0.8, 0.7, 0.6, 0.5]]))
    bn.set_node('c', np.array([3]))
    xs, dist = bn.get_node('a')

    pre_scale, post_scale, bias = bn._linearize('a', dist)
    test_results(dist, bn, pre_scale, xs, post_scale, bias)

    # 2d, tri-linear case
    bn.set_node('a', np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.9], [0.8, 0.7], [0.6, 0.5]]),
                ("exp(-dot(x,x)/2)", "2*b@a@c + 2 + c"))
    bn.set_node('b', np.array([[0.9, 0.8, 0.7, 0.6, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]]),
                ("exp(-dot(x,x)/2)", "2*b@a - 1"))
    bn.set_node('c', np.array([[1, 2], [3, 4]]),
                ("exp(-dot(x,x)/2)", "2*b@a@c + 2*c"))

    xs, dist = bn.get_node('a')
    pre_scale, post_scale, bias = bn._linearize('a', dist)
    test_results(dist, bn, pre_scale, xs, post_scale, bias)
    xs, dist = bn.get_node('b')
    pre_scale, post_scale, bias = bn._linearize('b', dist)
    test_results(dist, bn, pre_scale, xs, post_scale, bias)
    xs, dist = bn.get_node('c')
    pre_scale, post_scale, bias = bn._linearize('c', dist)
    test_results(dist, bn, pre_scale, xs, post_scale, bias)


if __name__ == '__main__':
    test_linearize()
