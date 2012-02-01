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


grads = {}

class PDDBM(Model):

    def __init__(self,
            s3c,
            dbm
            ):

        super(PDDBM,self).__init__()

        self.s3c = s3c
        s3c.e_step.autonomous = False

        self.dbm = dbm

        self.rng = np.random.RandomState([1,2,3])

        self.s3c.bias_hid = self.dbm.bias_vis

        self.nvis = s3c.nvis


        self.num_g = len(self.dbm.W)

        self.dbm.redo_everything()


        for param in self.get_params():
            grads[param] = sharedX(np.zeros(param.get_value().shape))

        self.test_batch_size = 2

        params_to_approx_grads = self.dbm.get_neg_phase_grads()

        updates = {}

        for param in grads:
            if param in params_to_approx_grads:
                updates[grads[param]] = params_to_approx_grads[param]
            else:
                updates[grads[param]] = T.zeros_like(param)

        sampling_updates = self.dbm.get_sampling_updates()

        for key in sampling_updates:
            assert key not in updates
            updates[key] = sampling_updates[key]

        print 'compiling reset grad func'
        global f
        f = function([], updates = updates)

    def get_params(self):
        return list(set(self.s3c.get_params()).union(set(self.dbm.get_params())))

model = PDDBM(
        dbm = DBM (
                negative_chains = 100,
                monitor_params = 1,
                rbms = [ serial.load("/u/goodfeli/galatea/pddbm/config/stl/full/layer_2_from_C1_A.pkl") ]
        ),
        s3c =  serial.load("/u/goodfeli/galatea/pddbm/config/stl/full/layer_1_C1.pkl"),
)

before =  theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
f()
gc.collect(); gc.collect(); gc.collect()
after = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
assert after[0] >= before[0]

