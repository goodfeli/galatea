#script to demonstrate that theano leaks memory on the gpu

import numpy as np
#from pylearn2.config import yaml_parse
from galatea.pddbm.pddbm import PDDBM
from galatea.pddbm import pddbm
from pylearn2.models.dbm import DBM
from pylearn2.utils import serial

model = PDDBM(
        learning_rate = .001,
        dbm = DBM (
                negative_chains = 100,
                monitor_params = 1,
                rbms = [ serial.load("/u/goodfeli/galatea/pddbm/config/stl/full/layer_2_from_C1_A.pkl") ]
        ),
        s3c =  serial.load("/u/goodfeli/galatea/pddbm/config/stl/full/layer_1_C1.pkl"),
       inference_procedure = pddbm.InferenceProcedure (
                schedule = [ ['s',1.],   ['h',1.],   ['g',0],   ['h', 0.4], ['s',0.4],
                             ['h',0.4], ['g',0],   ['h',0.4], ['s',0.4],  ['h',0.4],
                             ['g',0],   ['h',0.4], ['s',0.4], ['h', 0.4], ['g',0],
                             ['h',0.4], ['g',0],   ['h',0.4], ['s', 0.4], ['h',0.4] ],
                monitor_kl =  0,
                clip_reflections = 1,
                rho = 0.5
       ),
       print_interval =   100000,
       sub_batch = 1,
)

#model = yaml_parse.load(model_src)

X = np.random.RandomState([1,2,3]).randn(50,model.s3c.nvis)

model.learn_mini_batch(X)
