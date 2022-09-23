import numpy

# Define your own numpy-like object, for example cupy for GPU processing
np = numpy
# np = cupy

data_type = np.float64

# sp = scipy.sparse


def solve(a, b):
    return np.linalg.solve(a, b)
    # return sp.linalg.solve(sp.csr_matrix(a), b)


def rrms_error(a, b):
    return np.abs((a - b) / b)


verbose = False
