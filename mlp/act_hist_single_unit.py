from pylearn2.utils import serial
import sys
from matplotlib import pyplot
from pylearn2.config import yaml_parse
import theano.tensor as T
from theano import function
import numpy as np

ignore, model_path, layer, unit = sys.argv
layer_idx = int(layer)
unit_idx = int(unit)

model = serial.load(model_path)

dataset_yaml_src = model.dataset_yaml_src

dataset = yaml_parse.load(dataset_yaml_src)

input_space = model.get_input_space()

X = input_space.make_theano_batch()

outputs = []
below = X
for i, layer in enumerate(model.layers):
    above = layer.fprop(below)
    below = above
    if i == layer_idx:
        assert hasattr(layer, 'pool_size')
        outputs.append(above[:,unit_idx])
assert len(outputs) > 0

f = function([X], outputs)

X = dataset.X

batch_size = 100

act = []

for i in xrange(0, X.shape[0], batch_size):
    batch = X[i:i+batch_size,:]
    batch_act = f(batch)
    batch_act = np.concatenate([elem.reshape(elem.size) for elem in batch_act],axis=0)
    act.append(batch_act)
act = np.concatenate(act, axis=0)

print act.shape
pyplot.hist(act, bins=1000)
print 'Showing...'
pyplot.show()
