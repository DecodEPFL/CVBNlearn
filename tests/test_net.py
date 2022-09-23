# Copyright JSB 2022 (jean-sebastien.brouillon@epfl.ch)

import pandas as pd
from conf import np
import random
import string

"""
#
#   Tests Tests for edition features of Bayesian network class
#
"""

from src.bayesian_net import CVBNet
from src.constrained_functional import ConstrainedDistribution


def test_from_pandas():
    bn = CVBNet()

    df = pd.DataFrame(dtype=object)
    df["A"] = [0, 1]
    df["B"] = [2, 3]

    bn.from_pandas(df, pd.DataFrame(dtype=object))

    assert(list(bn.data.keys()) == ["A", "B"])
    assert(np.all(bn.data["A"] == np.array([0, 1])))
    assert(np.all(bn.data["B"] == np.array([2, 3])))

    return bn, df


def test_to_pandas():
    bn, df = test_from_pandas()
    dfr, _ = bn.to_pandas()

    assert((dfr == df).all().all())


def test_from_to_pandas():
    bn = CVBNet()
    n = 20
    names = random.sample(string.ascii_letters, k=7)

    df = pd.DataFrame(dtype=object)
    df[names[0]] = np.random.normal(size=(n, 3)).tolist()
    df[names[1]] = np.random.normal(size=(n, 5)).tolist()
    df[names[2]] = np.random.normal(size=(n, 8)).tolist()
    df[names[3]] = np.random.normal(size=(n, 12)).tolist()

    p = pd.DataFrame(dtype=object)
    p[names[4]] = np.random.normal(size=(1, 2)).tolist()
    p[names[5]] = np.random.normal(size=(1, 5)).tolist()
    p[names[6]] = np.random.normal(size=(1, 1)).tolist()

    bn.from_pandas(df, p)
    dfr, pr = bn.to_pandas()

    assert((dfr == df).all().all())
    assert((pr == p).all().all())

    return bn


def test_set_get_node_value():
    bn = test_from_to_pandas()
    data = np.random.normal(size=(bn.N, 2))

    bn.set_node("abc", data)
    _data, _ = bn.get_node("abc")

    assert(np.all(data == _data))


def test_set_get_node_distribution():
    bn = test_from_to_pandas()
    node_names = (list(bn.data.keys())[0], list(bn.data.keys())[1])

    dist = ConstrainedDistribution()
    dist.set_function(lambda x: x*x, lambda x: x[node_names[0]]*x[node_names[1]])
    xs = {node_names[0]: np.random.normal(), node_names[1]: np.random.normal()}

    bn.set_node("abc", None, dist)
    _, _dist = bn.get_node("abc")
    assert(np.all(dist(xs) == _dist(xs)))

    bn.set_node("abc", np.array([2]), ConstrainedDistribution(("-x*x", "abc/4")))
    _, _dist = bn.get_node("abc")
    assert(-(2/4)**2 == _dist(bn.data)[0])


test_from_pandas()
test_to_pandas()
test_from_to_pandas()
test_set_get_node_value()
test_set_get_node_distribution()
