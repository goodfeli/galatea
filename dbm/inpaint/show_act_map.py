m = 1
from pylearn2.utils import serial
from pylearn2.utils import sharedX
from pylearn2.config import yaml_parse
from theano import function
import sys
import numpy as np
from pylearn2.gui.patch_viewer import PatchViewer

ignore, model_path, layer_str = sys.argv

layer = int(layer_str)

model = serial.load(model_path)

model.set_batch_size(m)

if hasattr(model,'dataset_yaml_src'):
    dataset = yaml_parse.load(model.dataset_yaml_src)
else:
    from pylearn2.datasets.cifar10 import CIFAR10
    dataset = CIFAR10(which_set = 'test', gcn = 55.)

X = sharedX(dataset.get_batch_topo(m))

H_hat = model.mf(X)

H_hat = H_hat[layer]

p, h = H_hat

f = function([],[p,h])

while True:
    p,h = f()
    print 'p range: ',(p.min(),p.max())
    print 'h range: ',(h.min(),h.max())
    print 'p mean: ',p.mean()
    print 'h mean: ',h.mean()
    print 'p shape: ',p.shape
    print 'h shape: ',h.shape
    assert p.shape[0] == 1
    assert h.shape[0] == 1

    n = p.shape[-1]

    c = int(np.sqrt(2*n))
    if c % 2 != 0:
        c += 1
    r = 2 * n / c
    if r * c < 2 * n:
        r += 1

    pv = PatchViewer( (r,c), (h.shape[1],h.shape[2]), is_color = False )

    for i in xrange(n):
        hi = h[0,:,:,i]
        pv.add_patch(hi*2.-1.,rescale=False)
        pv.add_patch(p[0,:,:,i]*2.-1.,rescale=False)


    pv.show()

    x = X.get_value()[0,:,:,:]
    pv = PatchViewer( (1,3), (x.shape[0],x.shape[1]), is_color = x.shape[-1] == 3)

    pv.add_patch(dataset.adjust_for_viewer(x),rescale=False)

    origin = model.visible_layer.space.get_origin()
    mu = origin + model.visible_layer.mu.get_value()
    print 'mu range: ',(mu.min(),mu.max())
    pv.add_patch(mu,rescale=True)
    beta = origin + model.visible_layer.beta.get_value()
    print 'beta range: ',(beta.min(),beta.max())
    beta -= beta.min()
    beta /= max(1e-9,beta.max())
    beta = beta * 2 -1
    pv.add_patch(beta,rescale=False)

    pv.show()


    print 'waiting...'
    x = raw_input()

    if x == 'q':
        quit()

    print 'running...'

    X.set_value(dataset.get_batch_topo(m).astype(X.dtype))


