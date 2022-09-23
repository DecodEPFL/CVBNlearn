# Copyright JSB 2022 (jean-sebastien.brouillon@epfl.ch)

import numpy as np
from src.functional import ConditionalDistribution


class ConstrainedDistribution:
    """
    #
    #   Class for functional object restricted to the form "outer o inner (x)"
    #   The outer function should be a unimodal probability distribution p(x) with the mode at x = 0
    #   The inner function should be multi-affine with the node names as variables
    #
    """

    def __init__(self, other=None, is_distribution_log=True):
        """
        Initialize the functional object as the placeholder x -> 1 (with the same shape)

        Warning: the inner function is expressed as a combination of several variables x['variables_name'],
        or simply "variable_name" in a str expression. They are combined into a single result denoted by "x".
        The outer function is expressed depending ONLY on "x". For example, as lambda x: x.

        :param other: other ConstrainedDistribution to copy or
        tuple of ConditionalDistribution as (outer, inner).
        :param is_distribution_log: Flag that is True if the self._outer contains log(p) instead of p
        """
        self._inner = ConditionalDistribution()
        self._outer = ConditionalDistribution()
        self._is_log = is_distribution_log

        if other is not None:
            if type(other) is ConstrainedDistribution:
                (self._inner, self._outer) = other.get_function()
            if type(other) is tuple and len(other) == 2:
                if type(other[0]) is str and type(other[1]) is str:
                    self._inner.set_expression(other[1])
                    self._outer.set_expression(other[0], False)
                else:
                    self._inner.set_function(other[1])
                    self._outer.set_function(other[0])
            else:
                raise TypeError("ConstrainedDistribution can only copy objects of the same type \
                or a tuple of two ConditionalDistribution objects, lambda expressions, or str expressions.")

    def __call__(self, x):
        """
        Operator () to obtain the probability distribution as distribution(x),
        for the realization x given.

        :param x: dictionary of values
        """

        return self._outer(self._inner(x))

    def residual(self, x):
        r = self._inner(x)
        return np.dot(r.T, r)

    def is_log(self):
        return self._is_log

    def set_function(self, outer, inner, is_distribution_log=True):
        """
        Gives the functional objects for the inner and outer functions

        :param outer: lambda expression or ConditionalDistribution the outer function
        :param inner: lambda expression or ConditionalDistribution for the inner function
        :param is_distribution_log: Flag that is True if the self._outer contains log(p) instead of p
        """

        self._inner.set_function(inner)
        self._outer.set_function(outer)
        self._is_log = is_distribution_log
        return self

    def get_function(self):
        """
        Gives the functional objects for the inner and outer functions

        :return: tuple of two ConditionalDistribution objects
        """
        return self._outer, self._inner

    def set_expression(self, outer_xpr, inner_xpr, is_distribution_log=True):
        """
        Sets the inner and outer functions from an expresion written in a string.
        see ConditionalDistribution.set_expression for more information

        :param outer_xpr: string expression for the outer function
        :param inner_xpr: string expression for the inner function
        :param is_distribution_log: Flag that is True if the self._outer contains log(p) instead of p
        """

        self._inner.set_expression(inner_xpr)
        self._outer.set_expression(outer_xpr, False)
        self._is_log = is_distribution_log
        return self
