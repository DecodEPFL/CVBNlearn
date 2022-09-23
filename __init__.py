from src.bayesian_net import CVBNet

def create_empty_net():
    return CVBNet()

def create_net_from_pandas(samples, parameters, distributions=None):
    """
    see CVBNet.from_pandas.

    :param samples: dataframe of time-variant variables
    :param parameters: dataframe of time-invariant parameters
    :param distributions: (optional) dictionary containing all the distribution functionnals
    :return: net
    """
    net = CVBNet()
    net.from_pandas(samples, parameters, distributions)
    return net

def create_net_from_dict(variables, distributions):
    """
    see CVBNet.set_node.

    :param variables: dictionary containing the values of variables or nan if unknown
    :param distributions: (optional) dictionary containing all the distribution functionnals
    :return: net
    """

    net = CVBNet()
    for key, value in variables:
        net.set_node(key, value, distributions[key])


