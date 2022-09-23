# Copyright JSB 2022 (jean-sebastien.brouillon@epfl.ch)

import numpy as np

"""
#
#   Tests for functional object based on lambda expression
#
"""

from src.functional import ConditionalDistribution
from src.constrained_functional import ConstrainedDistribution


def test_func_constructor():
    dist = ConditionalDistribution()
    assert(dist(2) == 1)
    assert((dist(np.array([3, 2, 0, 5])) == np.array([1, 1, 1, 1])).all())


def test_set_from_lambda():
    dist = ConditionalDistribution()
    xs = {'a': np.random.normal(), 'b': np.random.normal(),
          'c': np.random.normal(), 'd': np.random.normal()}
    dist.set_function(lambda x: x['a']*x['b'])
    assert(dist(xs) == xs['a']*xs['b'])

    dist.set_function(lambda x: np.exp(x['a']+x['b']*x['c']))
    assert(dist(xs) == np.exp(xs['a'] + xs['b']*xs['c']))

    dist.set_function(lambda x: x['d']*np.exp(x['a']+np.power(np.abs(x['b']*x['c']), 0.5)))
    assert(dist(xs) == xs['d']*np.exp(xs['a']+np.power(np.abs(xs['b']*xs['c']), 0.5)))


def test_set_from_expression():
    dist = ConditionalDistribution()
    xs = {'a': np.random.normal(), '_b': np.random.normal(),
          'c': np.random.normal(), 'd': np.random.normal()}
    dist.set_expression("a*_b")
    assert(dist(xs) == xs['a']*xs['_b'])

    dist.set_expression("exp(a + _b*c)")
    assert(dist(xs) == np.exp(xs['a'] + xs['_b']*xs['c']))

    dist.set_expression("d*np.exp(a + np.abs(_b*c)^0.5)")
    assert(dist(xs) == xs['d']*np.exp(xs['a']+np.power(np.abs(xs['_b']*xs['c']), 0.5)))


test_func_constructor()
test_set_from_lambda()
test_set_from_expression()
