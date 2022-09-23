# CVBNlearn
Inference tool for continuous variables Bayesian Networks
copy package in project or venv to use it.

minimal code:
``` python script
import numpy as np
import CVBNlearn as cvbn

# Create network
net = cvbn.create_empty_net()

# Add a node
net.set_node('node_a', np.zeros(10, 1) ,("x ** 2", "node_a - 1"))
# Parameters are: name of the node, realizations of the random variable,
# unimodal conditional distribution of a multilinear function of self and parent nodes

# Add other nodes in the same way

# Node with data to infer uses np.nan, pd.NA or None array
net.set_node('node_b', np.zeros(10, 1)*np.nan ,("x ** 2", "node_a - node_b"))

# Set an initial guess
guess = {'node_a' : np.zeros(10, 1), 'node_b' : np.zeros(10, 1)}

# Perform the inference
bn.infer(guess)

# Extract the results
print(bn.data)
```
