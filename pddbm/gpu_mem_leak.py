#script to demonstrate that theano leaks memory on the gpu

import numpy as np
from pylearn2.models.dbm import DBM
from pylearn2.utils import serial
from pylearn2.models.model import Model
from pylearn2.utils import as_floatX
from pylearn2.utils import sharedX
import warnings
import theano.tensor as T
from theano import config
from theano import function
from theano.gof.op import get_debug_values
from pylearn2.models.s3c import reflection_clip
from pylearn2.models.s3c import damp
from pylearn2.models.s3c import S3C
from pylearn2.models.s3c import SufficientStatistics
from galatea.pddbm.pddbm import flatten
import time
import theano
import gc

#rbm =  serial.load("/u/goodfeli/galatea/pddbm/config/stl/full/layer_2_from_C1_A.pkl")

V_chain = sharedX(np.zeros((100,8478)))
H_chain = sharedX(np.zeros((100,400)))


W = sharedX(np.zeros((8478,400)))
b = sharedX(np.zeros((400,)))
c = sharedX(np.zeros((8478,)))


grads = {}

params= [W]

for param in params:
    grads[param] = sharedX(np.zeros(param.get_value().shape))

v = T.mean(V_chain, axis=0)

v_bias_contrib = T.dot(v, c)

exp_vh = T.dot(V_chain.T,H_chain)

v_weights_contrib = T.sum(W)

total = v_bias_contrib + v_weights_contrib

highest_bias_contrib = T.dot(T.mean(H_chain,axis=0), b)

obj = total + highest_bias_contrib

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

