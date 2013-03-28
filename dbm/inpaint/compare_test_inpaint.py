import sys

model_paths = sys.argv[1:]

from pylearn2.utils import serial

print 'Loading models...'
models = map(serial.load, model_paths)

from pylearn2.datasets.binarizer import Binarizer
from pylearn2.datasets.mnist import MNIST

print 'Loading data...'
raw = MNIST(which_set='test', one_hot=True)
train = Binarizer(raw)

print 'Compiling cost functions...'
for model in models:
    model.niter = 10

from galatea.dbm.inpaint.super_inpaint import SuperInpaint
from galatea.dbm.inpaint.super_inpaint import MaskGen
from pylearn2.utils import sharedX

mask_gen = MaskGen(
    drop_prob = sharedX(0.1),
    balance = 0,
    sync_channels = 0
)

cost = SuperInpaint(
        both_directions = 0,
        noise = 0,
        supervised = 1,
        mask_gen = mask_gen
)

from pylearn2.utils import function

def get_obj_func(model):
    X = model.get_input_space().make_batch_theano()
    Y = model.get_output_space().make_batch_theano()
    obj = cost(model, X, Y)
    return function([X,Y], obj)

funcs = map(get_obj_func, models)

# Set up main loop
from pylearn2.utils import safe_izip

def get_objs():
    n = 0.
    aves = [0. for model in models]
    m = 0
    for X, Y in train.iterator(batch_size = 5000, mode='sequential', targets=True):
        objs = [func(X, Y) for func in funcs]
        n += 1.
        aves = [ave + (obj - ave) / n for ave, obj in safe_izip(aves, objs)]
        m += X.shape[0]
    if m != 10000:
        raise AssertionError(str(m))
    return aves

import numpy as np

x = []
ys = []

eps = .01
for drop_prob in np.arange(eps, 1.0, eps):
    print 'Running experiment for drop_prob=%f...' % drop_prob
    mask_gen.drop_prob.set_value(np.cast[mask_gen.drop_prob.dtype](drop_prob))
    x.append(drop_prob)
    ys.append(get_objs())

save_path = 'test_comp.pkl'
print 'Saving to %s...' % save_path
serial.save(save_path, {'x': x, 'ys': ys, 'model_paths' : model_paths})
print 'Done.'
