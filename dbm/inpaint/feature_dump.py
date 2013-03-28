from pylearn2.utils import serial
from pylearn2.utils import sharedX
from pylearn2.config import yaml_parse
from theano import function
import sys
import numpy as np

ignore, model_path, layer_str, output_path = sys.argv

layer = int(layer_str)

model = serial.load(model_path)

# Get access to the intermediate layers of the augmented DBM
if hasattr(model, 'super_dbm'):
    model = model.super_dbm

batch_size = 25
model.set_batch_size(batch_size)

dataset = yaml_parse.load(model.dataset_yaml_src)
assert dataset.X.shape[0] == 50000

X = sharedX(dataset.get_batch_topo(batch_size))

layer_obj = model.hidden_layers[layer]
p_space = layer_obj.output_space
n = p_space.shape[0] * p_space.shape[1] * p_space.nchannels

num_examples = dataset.X.shape[0]
features = np.zeros((num_examples, n), dtype='float32')

H_hat = model.mf(X)

H_hat = H_hat[layer]

p, h = H_hat

feat_th = p.reshape((batch_size, n))

f = function([], feat_th)

for i in xrange(0, num_examples, batch_size):
    print i
    batch = dataset.X[i:i+batch_size, :].astype(X.dtype)
    batch = dataset.get_topological_view(batch)
    X.set_value(batch)
    features[i:i+batch_size, :] = f()

serial.save(output_path, features)
