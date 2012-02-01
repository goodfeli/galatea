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


obj = dbm.expected_energy(V_hat = dbm.V_chains, H_hat = dbm.H_chains)

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

