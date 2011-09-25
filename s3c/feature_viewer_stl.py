import warnings
import numpy as np
from pylearn2.datasets.cifar10 import CIFAR10
from theano import config
from theano import tensor as T
#from theano.sandbox.neighbours import images2neibs
from theano import function
from pylearn2.datasets.preprocessing import ExtractPatches, ExtractGridPatches, ReassembleGridPatches
from pylearn2.gui.patch_viewer import make_viewer
from pylearn2.utils import serial
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter
from pylearn2.gui.patch_viewer import PatchViewer


batch_start = 0
batch_size = 15
k = 15

import sys
model_path = sys.argv[1]

print 'loading model'
model = serial.load(model_path)

print 'loading dataset'
dataset = serial.load('${PYLEARN2_DATA_PATH}/stl10_32x32/train.pkl')
X = dataset.get_design_matrix()[batch_start:batch_start + batch_size,:]

size = np.sqrt(model.nvis/3)


pv1 = make_viewer( X/127.5, is_color = True, rescale = False)
dataset.set_design_matrix(X)

patchifier = ExtractGridPatches( patch_shape = (size,size), patch_stride = (1,1) )


if size ==6:
    pipeline = serial.load('${PYLEARN2_DATA_PATH}/stl10_patches/preprocessor.pkl')
else:
    print size
    assert False

assert isinstance(pipeline.items[0], ExtractPatches)
pipeline.items[0] = patchifier

print 'applying preprocessor'
dataset.apply_preprocessor(pipeline, can_fit = False)


X2 = dataset.get_design_matrix()



print 'defining features'
V = T.matrix()
model.make_Bwp()
d = model.e_step.mean_field(V = V)

H = d['H']
Mu1 = d['Mu1']

warnings.warn('using MAP')
feat = (H > 0.5) * Mu1

print 'compiling theano function'
f = function([V],feat)

print 'running theano function'
feat = f(X2)

feat_dataset = DenseDesignMatrix(X = feat, view_converter = DefaultViewConverter([1, 1, feat.shape[1]] ) )

print 'reassembling features'
ns = 32 - size + 1
depatchifier = ReassembleGridPatches( orig_shape  = (ns, ns), patch_shape=(1,1) )
feat_dataset.apply_preprocessor(depatchifier)

print 'making topological view'
topo_feat = feat_dataset.get_topological_view()
assert topo_feat.shape[0] == X.shape[0]

print 'assembling visualizer'

n = np.ceil(np.sqrt(model.nhid))

pv3 = PatchViewer(grid_shape = (X.shape[0], k), patch_shape=(ns,ns), is_color= False)
pv4 = PatchViewer(grid_shape = (n,n), patch_shape = (size,size), is_color = True, pad = (7,7))

idx = sorted(range(model.nhid), key = lambda l : -topo_feat[:,:,:,l].std() )

W = model.W.get_value()

weights_view = dataset.get_weights_view( W.T )

p_act = 1. / (1. + np.exp(- model.bias_hid.get_value()))
p_act /= p_act.max()

mu_act = np.abs(model.mu.get_value())
mu_act /= mu_act.max()
mu_act += 0.5

alpha_act = model.alpha.get_value()
alpha_act /= alpha_act.max()

for j in xrange(model.nhid):
    cur_idx = idx[j]

    cur_p_act = p_act[cur_idx]
    cur_mu_act = mu_act[cur_idx]
    cur_alpha_act = alpha_act[cur_idx]

    activation = (cur_p_act, cur_mu_act, cur_alpha_act)

    pv4.add_patch(weights_view[cur_idx,:], rescale = True,
            activation = activation)

for i in xrange(X.shape[0]):
    #indices of channels sorted in order of descending standard deviation on this example
    #plot the k most interesting channels
    for j in xrange(k):
        pv3.add_patch(topo_feat[i,:,:,idx[j]], rescale = True, activation = 0.)

pv1.show()
pv3.show()
pv4.show()

#pv2 = make_viewer(X2, is_color = True)
#pv2.show()
