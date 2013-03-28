import sys

_, model_path = sys.argv

from pylearn2.utils import serial

model = serial.load(model_path)

from pylearn2.config import yaml_parse

dataset = yaml_parse.load(model.dataset_yaml_src)

batch = model.get_input_space().make_batch_theano()

state_below = batch
outputs = []
for layer in model.layers[:-1]:
    state_below, counts = layer.foo(state_below)
    outputs.append(counts)

from theano import function
f = function([batch], outputs)

counts = []

for i in xrange(0,40000,100):
    batch = dataset.X[i:i+100,:]

    new = f(batch)

    if counts:
        counts = [count + new_counts for count, new_counts in zip(counts, new)]
    else:
        counts = new

print "Proportion equal to 0: "
mats = []
for count in counts:
    print (count == 0).mean()
    mats.append(count.reshape(count.size))

import numpy as np
mats = np.concatenate(mats, axis=0)

from matplotlib import pyplot
pyplot.hist(mats, bins=100)
pyplot.show()
