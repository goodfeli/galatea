from pylearn2.utils import serial
from pylearn2.utils import sharedX
from pylearn2.config import yaml_parse
from theano import function
import sys
import numpy as np
from pylearn2.gui.patch_viewer import PatchViewer

m = 25
ignore, model_path, layer_str = sys.argv

layer = int(layer_str)

model = serial.load(model_path)

# Get access to the intermediate layers of the augmented DBM
if hasattr(model, 'super_dbm'):
    model = model.super_dbm

model.set_batch_size(m)

if hasattr(model,'dataset_yaml_src'):
    dataset = yaml_parse.load(model.dataset_yaml_src)
else:
    from pylearn2.datasets.cifar10 import CIFAR10
    dataset = CIFAR10(which_set = 'test', gcn = 55.)

ex = 0
X = sharedX(dataset.get_topological_view(dataset.X[ex:ex+m,:]))

H_hat = model.mf(X)

H_hat = H_hat[layer]

H_hat

p, h = model.hidden_layers[layer].state_to_b01c(H_hat)
del h

f = function([],p)

while True:
    p = f()
    print 'p range: ',(p.min(),p.max())
    mins = p.min(axis=0)
    print 'p min: ', (mins.min(), mins.mean(), mins.max())
    print 'p mean: ',p.mean()
    print 'p shape: ',p.shape
    assert p.shape[0] == m

    n = p.shape[-1]

    pv = PatchViewer( (m,n), (p.shape[1],p.shape[2]), is_color = False )

    for j in xrange(m):
        draw = 1.
        for i in xrange(n):
            pv.add_patch(draw * p[j,:,:,i]*2.-1.,rescale=False)
            if p[:,:,:,i].min(axis=0).max() > .9:
                draw = 0.

    pv.show()



    print 'waiting...'
    x = raw_input()

    if x == 'q':
        quit()

    print 'running...'

    ex += m
    X.set_value(dataset.get_topological_view(
        dataset.X[ex:ex+m,:]).astype(X.dtype))


