#script to demonstrate that theano leaks memory on the gpu

import numpy as np
from pylearn2.utils import sharedX
from theano import function
import theano
import gc

W = sharedX(np.zeros((400,400)))

grad  = sharedX(np.zeros(W.get_value().shape))

updates = { grad : W}

f = function([], updates = updates)

before =  theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
f()
gc.collect(); gc.collect(); gc.collect()
after = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
assert after[0] >= before[0]

