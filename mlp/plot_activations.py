from pylearn2.utils import serial
import sys
from matplotlib import pyplot
from pylearn2.config import yaml_parse
import theano.tensor as T
from theano import function
import numpy as np

ignore, model_path, layer_idx, unit_idx = sys.argv
layer_idx = int(layer_idx)
unit_idx = int(unit_idx)

model = serial.load(model_path)

dataset_yaml_src = model.dataset_yaml_src

dataset = yaml_parse.load(dataset_yaml_src)

input_space = model.get_input_space()

X = input_space.make_theano_batch()

below = X
for i, layer in enumerate(model.layers):
    above = layer.fprop(below)
    if i == layer_idx:
        break
    below = above
assert layer is model.layers[layer_idx]

assert layer.pool_size == 2
assert layer.pool_stride == 2
assert unit_idx < layer.detector_layer_dim // 2

W, = layer.transformer.get_params()
i = unit_idx
w1 = W[:, 2*i]
w2 = W[:, 2*i+1]
d1 = w1 / T.sqrt(T.sqr(w1).sum())
d2 = w2 / T.sqrt(T.sqr(w2).sum())
d2 = d2 - T.dot(d1, d2) * d1
d2 = d2 / T.sqrt(T.sqr(d2).sum())


p1 = T.dot(below, d1)
p2 = T.dot(below, d2)

act = above[:,i]

f = function([X], [p1, p2, act])

X = dataset.X

batch_size = 100

p1 = []
p2 = []
act = []

for i in xrange(0, X.shape[0], batch_size):
    batch = X[i:i+batch_size,:]
    batch_p1, batch_p2, batch_act = f(batch)
    p1.append(batch_p1)
    p2.append(batch_p2)
    act.append(batch_act)

p1 = np.concatenate(p1, axis=0)
p2 = np.concatenate(p2, axis=0)
act = np.concatenate(act, axis=0)

pyplot.scatter(p1, p2, c=act, edgecolors='none')
print 'Showing...'
pyplot.show()
