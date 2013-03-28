import sys

_, model_path = sys.argv

from pylearn2.utils import serial
import numpy as np

model = serial.load(model_path)

from pylearn2.config import yaml_parse

dataset = yaml_parse.load(model.dataset_yaml_src)

batch = model.get_input_space().make_batch_theano()
states = model.fprop(batch, return_all=True)
max_states = [state.max(axis=0) for state in states]

from theano import function
f = function([batch], max_states)

maxes = []

for i in xrange(0,40000,100):
    batch = dataset.X[i:i+100,:]

    max_states = f(batch)

    if maxes:
        maxes = [np.maximum(mx, mxs) for mx, mxs in zip(maxes, max_states)]
    else:
        maxes = max_states

print "Proportion less than 0: "
for mx in maxes:
    print (mx < 0.).mean()

