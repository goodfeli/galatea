from pylearn2.utils import serial
import sys
from matplotlib import pyplot
from pylearn2.config import yaml_parse
import theano.tensor as T
from theano import function
import numpy as np

ignore, model_path = sys.argv

model = serial.load(model_path)

dataset_yaml_src = model.dataset_yaml_src

dataset = yaml_parse.load(dataset_yaml_src)

input_space = model.get_input_space()

X = input_space.make_theano_batch()
if X.ndim > 2:
    assert False # doesn't support topo yet

outputs = []
below = X
for layer in model.layers:
    above = layer.fprop(below)
    below = above
    if hasattr(layer, 'pool_size') or hasattr(layer, 'pool_shape'):
        outputs.append(above)
assert len(outputs) > 0

f = function([X], outputs)

X = dataset.X

batch_size = 100
max_batches = 1

act = []

for i in xrange(0, min(batch_size * max_batches, X.shape[0]), batch_size):
    batch = X[i:i+batch_size,:]
    batch_act = f(batch)
    batch_act = np.concatenate([elem.reshape(elem.size) for elem in batch_act],axis=0)
    act.append(batch_act)
act = np.concatenate(act, axis=0)

pyplot.hist(act, bins=10000)
print 'Showing...'
pyplot.show()
