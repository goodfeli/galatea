import numpy as np
from pylearn2.utils import sharedX
from pylearn2.expr.probabilistic_max_pooling import max_pool_c01b

X = sharedX(np.zeros((16, 34, 34, 2)))

P, H = max_pool_c01b(X, pool_shape=[2, 2])

obj = P.sum() + H.sum()

from theano import tensor as T
from theano import function

g = T.grad(obj, X)

f = function([], g)
