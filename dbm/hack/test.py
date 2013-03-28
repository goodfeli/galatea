import numpy as np
from scipy import io
from pylearn2.models import dbm
from pylearn2.utils import sharedX
from theano import function
from theano import config
from galatea.dbm.inpaint.super_dbm import SuperDBM

matlab = io.loadmat('pylearn2_test_data.mat')

for key in matlab:
    try:
        matlab[key] = matlab[key].astype(config.floatX)
    except:
        pass

data = matlab['data']
targets = matlab['targets']
vishid = matlab['vishid']
hidbiases = matlab['hidbiases']
visbiases = matlab['visbiases']
hidpen = matlab['hidpen']
penbiases = matlab['penbiases']
labpen = matlab['labpen']
targetout = matlab['targetout']
correct = matlab['correct']
labbiases = matlab['labbiases']
poshidprobs = matlab['poshidprobs']
pospenprobs = matlab['pospenprobs']

batch_size, num_vis = data.shape
_, num_hid = vishid.shape
assert _ == num_vis
assert hidbiases.shape == (1, num_hid)
hidbiases = hidbiases[0,:]
assert visbiases.shape == (1, num_vis)
visbiases = visbiases[0,:]
_, num_pen = hidpen.shape
assert _ == num_hid
assert penbiases.shape == (1, num_pen)
penbiases = penbiases[0,:]
num_lab, _ = labpen.shape
assert _ == num_pen
assert targetout.shape == (batch_size, num_lab)
assert correct == batch_size
assert labbiases.shape == (1, num_lab)
labbiases = labbiases[0,:]

vis = dbm.BinaryVector(num_vis)
hid = dbm.BinaryVectorMaxPool(detector_layer_dim=num_hid, pool_size=1, layer_name='hid', irange=1.)
pen = dbm.BinaryVectorMaxPool(detector_layer_dim=num_pen, pool_size=1, layer_name='pen', irange=1.)
lab = dbm.Softmax(n_classes=num_lab, layer_name='lab', irange=1.)

model = SuperDBM(niter=11, batch_size=batch_size,
        visible_layer=vis,
        hidden_layers=[hid,pen,lab])

vis.set_biases(visbiases)
hid.set_biases(hidbiases)
hid.set_weights(vishid)
pen.set_biases(penbiases)
pen.set_weights(hidpen)
lab.set_weights(labpen.T)
lab.set_biases(labbiases)

V = sharedX(data)
Y = sharedX(targets)

Q = model.mf(V=V, Y=Y)

H1, H2, _ = Q
H1, H1p = H1
assert H1 is H1p
H2, H2p = H2
assert H2 is H2p

f = function([], [H1, H2])

H1, H2 = f()

if not np.allclose(H1, poshidprobs):
    print "H1 does not match!"
    print "max diff: ",np.abs(H1-poshidprobs).max()
    assert False
assert np.allclose(H2, pospenprobs)


