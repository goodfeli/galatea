from pylearn2.datasets.mnist import MNIST
import numpy as np
one = MNIST(which_set='train', shuffle=True)
one_X = one.X.copy()
two = MNIST(which_set='train', start=0, stop=50000, shuffle=True)
two_X = two.X.copy()
three = MNIST(which_set='train', start=50000, stop=60000, shuffle=True)
three_X = three.X.copy()
assert np.all(one_X[0:50000,:] == two_X)
assert np.all(one_X[50000:,:] == three_X)
assert np.all(one_X == one.X)
assert np.all(two_X == two.X)
assert np.all(three_X == three.X)

