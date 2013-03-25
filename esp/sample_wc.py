#!/bin/env python
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
"""

Usage: python show_samples <path_to_a_saved_DBM.pkl>
Displays a batch of data from the DBM's training set.
Then interactively allows the user to run Gibbs steps
starting from that seed data to see how the DBM's MCMC
sampling changes the data.

"""

from pylearn2.utils import serial
import sys
from pylearn2.config import yaml_parse
import time
from theano import function
from theano.sandbox.rng_mrg import MRG_RandomStreams
import numpy as np
from pylearn2.expr.basic import is_binary

rows = 10
cols = 10
m = rows * cols

_, model_path = sys.argv

print 'Loading model...'
model = serial.load(model_path)
model.set_batch_size(m)


dataset_yaml_src = model.dataset_yaml_src

print 'Loading data (used for setting up visualization and seeding gibbs chain) ...'
dataset = yaml_parse.load(dataset_yaml_src)



if hasattr(model.visible_layer, 'beta'):
    beta = model.visible_layer.beta.get_value()
#model.visible_layer.beta.set_value(beta * 100.)
    print 'beta: ',(beta.min(), beta.mean(), beta.max())


x = 0


# Make shared variables representing the sampling state of the model
layer_to_state = model.make_layer_to_state(m)
# Seed the sampling with the data batch
vis_sample = layer_to_state[model.visible_layer]

def validate_all_samples():
    # Run some checks on the samples, this should help catch any bugs
    layers = [ model.visible_layer ] + model.hidden_layers

    def check_batch_size(l):
        if isinstance(l, (list, tuple)):
            map(check_batch_size, l)
        else:
            assert l.get_value().shape[0] == m


    for layer in layers:
        state = layer_to_state[layer]
        space = layer.get_total_state_space()
        space.validate(state)
        if 'DenseMaxPool' in str(type(layer)):
            p, h = state
            p = p.get_value()
            h = h.get_value()
            assert np.all(p == h)
            assert is_binary(p)
        if 'BinaryVisLayer' in str(type(layer)):
            v = state.get_value()
            assert is_binary(v)
        if 'Softmax' in str(type(layer)):
            y = state.get_value()
            assert is_binary(y)
            s = y.sum(axis=1)
            assert np.all(s == 1 )



validate_all_samples()

if x >= 0:
    if vis_sample.ndim == 4:
        vis_sample.set_value(vis_batch)
    else:
        vis_sample.set_value(dataset.get_batch_design(m))

validate_all_samples()

theano_rng = MRG_RandomStreams(2012+9+18)

if x > 0:
    sampling_updates = model.get_sampling_updates(layer_to_state, theano_rng,
            layer_to_clamp = { model.visible_layer : True }, num_steps = x)

    t1 = time.time()
    sample_func = function([], updates=sampling_updates)
    t2 = time.time()
    print 'Clamped sampling function compilation took',t2-t1
    sample_func()


# Now compile the full sampling update
sampling_updates = model.get_sampling_updates(layer_to_state, theano_rng)
assert layer_to_state[model.visible_layer] in sampling_updates

from pylearn2.utils import sharedX
word_mask = sharedX(np.zeros((4000,)))

vis = layer_to_state[model.visible_layer]
cur_mask = vis.max(axis=0)

import theano.tensor as T
sampling_updates[word_mask] = T.maximum(word_mask, cur_mask)


t1 = time.time()
sample_func = function([], word_mask.sum(), updates=sampling_updates)
t2 = time.time()

print 'Sampling function compilation took',t2-t1

i = 0
while True:
    print i, sample_func()
    i += 1



