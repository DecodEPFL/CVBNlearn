# Copyright JSB 2022 (jean-sebastien.brouillon@epfl.ch)

import re
from conf import np
# Need np for eval of functional


class ConditionalDistribution:
    """
    #
    #   Class for functional object based on lambda expression
    #
    """

    def __init__(self, other=None):
        """
        Initialize the functional object as the placeholder x -> 1 (with the same shape)

        :param other: other distribution or expression to copy
        """
        self._f = lambda x: {k: x[k]*0.0 + 1.0 for k in x} if isinstance(x, dict) else x*0.0 + 1.0
        self.xpr = None

        if other is not None:
            if type(other) is ConditionalDistribution:
                self._f = other.get_function()
                self.xpr = other.xpr
            elif type(other) is str:
                self.set_expression(other)
            else:
                raise TypeError("ConditionalDistribution can only copy objects of the same type or str.")

    def __call__(self, x):
        """
        Operator () to obtain the probability distribution as distribution(x),
        for the realization x given.

        :param x: dictionary of values
        """

        return self._f(x)

    def set_function(self, func):
        """
        Sets the function from an lambda expresion.
        Variables are referred to as x['variable_name'].
        For example, a*b + c is given as

        lambda x: x['a']*x['b'] + x['c']

        :param func: lambda expression
        """

        if type(func) is ConditionalDistribution:
            self._f = func.get_function()
        else:
            self._f = func
        return self

    def get_function(self):
        """
        Gives the lambda expression defining the functional object

        :return: lambda expression with variables as x['variable_name']
        """
        return self._f

    def set_expression(self, xpr, replace_variables=True):
        """
        Sets the function from an expresion written in a string.
        Variables x['variable_name'] will be inferred from the expression.
        All substrings containing only alphabetical characters and that are
        not in the following list will be considered variable names.
        If these names are wrong, KeyError exceptions may be raised during execution.

        List of known functions: exp, abs, sqrt.
        Note that if functions are written "np.func", for example "np.exp",
        they do not only contain alphabetical characters and will be recognized correctly

        :param xpr: string expression to read
        :param replace_variables: boolean to replace the node names by x['node_name'] or not
        """

        # Extract list of variables and functions
        variables_functions_list = list(set(filter(''.__ne__, re.split("[ +*@/-]|[()^]", xpr))))
        known_functions = ['exp', 'abs', 'sqrt', 'dot', 'log', 'sum', 'diag', 'transpose', 'min', 'max']

        # Replace variable names by x['variable_name'] and known functions by np.function
        for var in variables_functions_list:
            if var not in known_functions and bool(re.match("^[A-Za-z_]*$", var)):
                if replace_variables:
                    xpr = re.sub(r'\b' + var + r'\b', "x['" + var + "']", xpr)
            elif bool(re.match(r"[A-Za-z_]*$", var)):
                xpr = re.sub(r'\b' + var + r'\b', "np." + var, xpr)

        # Replace power sign to python one
        xpr = xpr.replace("^", "**")

        # Create lambda expression from str
        self._f = lambda x: eval(xpr)
        self.xpr = xpr
        return self
