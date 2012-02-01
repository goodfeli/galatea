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

dbm = DBM (
        negative_chains = 100,
        monitor_params = 1,
        rbms = [ serial.load("/u/goodfeli/galatea/pddbm/config/stl/full/layer_2_from_C1_A.pkl") ]
)

s3c =  serial.load("/u/goodfeli/galatea/pddbm/config/stl/full/layer_1_C1.pkl")

grads = {}

#s3c.e_step.autonomous = False
#rng = np.random.RandomState([1,2,3])

s3c.bias_hid = dbm.bias_vis

for param in list(set(s3c.get_params()).union(set(dbm.get_params()))):
    grads[param] = sharedX(np.zeros(param.get_value().shape))

m = dbm.V_chains.shape[0]

v = T.mean(dbm.V_chains, axis=0)

v_bias_contrib = T.dot(v, dbm.bias_vis)

exp_vh = T.dot(dbm.V_chains.T,dbm.H_chains[0]) / m

v_weights_contrib = T.sum(dbm.W[0] * exp_vh)

total = v_bias_contrib + v_weights_contrib

H_hat = dbm.H_chains
for i in xrange(len(H_hat) - 1):
    lower_H = H_hat[i]
    low = T.mean(lower_H, axis = 0)
    higher_H = H_hat[i+1]
    exp_lh = T.dot(lower_H.T, higher_H) / m
    lower_bias = dbm.bias_hid[i]
    W = dbm.W[i+1]

    lower_bias_contrib = T.dot(low, lower_bias)

    weights_contrib = T.sum( W * exp_lh) / m

    total = total + lower_bias_contrib + weights_contrib

highest_bias_contrib = T.dot(T.mean(H_hat[-1],axis=0), dbm.bias_hid[-1])

total = total + highest_bias_contrib

assert len(total.type.broadcastable) == 0

obj =  - total

constants = list(set(dbm.H_chains).union([dbm.V_chains]))

params = dbm.get_params()

agrads = T.grad(obj, params, consider_constant = constants)

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

