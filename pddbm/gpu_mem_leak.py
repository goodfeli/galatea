#script to demonstrate that theano leaks memory on the gpu

import numpy as np
from pylearn2.utils import sharedX
import theano.tensor as T
from theano import function
import theano
import gc



W = sharedX(np.zeros((8478,400)))


grads = {}

params= [W]

for param in params:
    grads[param] = sharedX(np.zeros(param.get_value().shape))

obj = T.sum(W)

agrads = T.grad(obj, params)

pags = {}

for param, grad in zip(params, agrads):
    pags[param] = grad

updates = {}

for param in grads:
    if param in pags:
        updates[grads[param]] = pags[param]
    else:
        updates[grads[param]] = T.zeros_like(param)

global f
f = function([], updates = updates)

before =  theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
f()
gc.collect(); gc.collect(); gc.collect()
after = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
assert after[0] >= before[0]

