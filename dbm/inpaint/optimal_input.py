from pylearn2.utils import serial
from pylearn2.utils import sharedX
from pylearn2.config import yaml_parse
import sys
import numpy as np
from pylearn2.optimization.batch_gradient_descent import BatchGradientDescent
from pylearn2.utils.image import show
from theano import tensor as T
from pylearn2.datasets import control

ignore, model_path, layer_idx, filter_idx = sys.argv
layer_idx = int(layer_idx)
filter_idx = int(filter_idx)

model = serial.load(model_path)

# Get access to the intermediate layers of the augmented DBM
if hasattr(model, 'super_dbm'):
    model = model.super_dbm

if hasattr(model,'dataset_yaml_src'):
    dataset = yaml_parse.load(model.dataset_yaml_src)
else:
    from pylearn2.datasets.cifar10 import CIFAR10
    dataset = CIFAR10(which_set = 'test', gcn = 55.)

max_elem = np.abs(dataset.X).max()
max_norm = np.square(dataset.X).sum(axis=1).max()
print 'max mag:',max_elem
print 'max norm: ',max_norm


batch_size = 1
model.set_batch_size(batch_size)
norm_penalty = .0001

#make act_func. should return num_layers tensors of shape batch_size, num_filters
print 'making act_func...'
X = sharedX(model.get_input_space().get_origin_batch(1))
H_hat = model.mf(X)
layer = model.hidden_layers[layer_idx]
state = H_hat[layer_idx]
p, h = state

p_shape = layer.get_output_space().shape
i = p_shape[0] / 2
j = p_shape[1] / 2

act = p[0,filter_idx,i,j]

obj = - act + norm_penalty * T.square(X).sum()

assert obj.ndim == 0

optimizer = BatchGradientDescent(objective = obj,
        params = [X],
        inputs = None,
        param_constrainers = None,
        max_iter = 1000,
        verbose = True,
        tol = None,
        init_alpha = (.001, .005, .01, .05, .1))

optimizer.minimize()

img = X.get_value()[0,:,:,:]

print 'max mag: ',np.abs(img).max()
print 'norm: ',np.square(img).sum()
print 'min: ',img.min()
print 'max: ',img.max()

img /= np.abs(img).max()

img *= .5
img += 1

show(img)
