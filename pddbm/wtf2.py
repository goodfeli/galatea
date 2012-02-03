#layer_1_C1 has a max p of .09 but mean h on stl10 train is .65
#what's going on?
#is it a difference between the posterior and the prior?
#is it a difference between unsup and train?
#is it a bug?

from pylearn2.utils import serial

model = serial.load('/u/goodfeli/galatea/pddbm/config/stl/full/layer_1_C1_cpu.pkl')

import theano.tensor as T

X = T.matrix()

model.make_pseudoparams()
obs = model.e_step.variational_inference(X)

from theano import function
f = function([X], obs['H_hat'])

from pylearn2.config import yaml_parse
dataset = yaml_parse.load(model.dataset_yaml_src)
#dataset = serial.load('${STL10_WHITENED_TRAIN}')

X = dataset.get_batch_design(100)

print f(X).mean(dtype='float64')
