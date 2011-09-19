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


batch_start = 0
batch_size = 15
k = 15

import sys
model_path = sys.argv[1]

print 'loading model'
model = serial.load(model_path)

print 'loading dataset'
dataset = CIFAR10(which_set = "train")
X = dataset.get_design_matrix()[batch_start:batch_start + batch_size,:]

size = np.sqrt(model.nvis/3)


pv1 = make_viewer( (X-127.5)/127.5, is_color = True, rescale = False)
dataset.set_design_matrix(X)

patchifier = ExtractGridPatches( patch_shape = (size,size), patch_stride = (1,1) )


if size == 8:
    pipeline = serial.load('${GOODFELI_TMP}/cifar10_preprocessed_pipeline_2M.pkl')
elif size ==6:
    pipeline = serial.load('${GOODFELI_TMP}/cifar10_preprocessed_pipeline_2M_6x6.pkl')
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

feat = H * Mu1

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


topo_feat_var = T.TensorType(broadcastable = (False,False,False,False), dtype=topo_feat.dtype)()
region_features = function([topo_feat_var],
        topo_feat_var.mean(axis=(1,2)) )

print "average pooling 2x2"
def average_pool( stride ):
    def point( p ):
        return p * ns / stride

    rval = np.zeros( (topo_feat.shape[0], stride, stride, topo_feat.shape[3] ) , dtype = X.dtype)

    for i in xrange(stride):
        for j in xrange(stride):
            rval[:,i,j,:] = region_features( topo_feat[:,point(i):point(i+1), point(j):point(j+1),:] )

    return rval


assert average_pool(2).shape == (X.shape[0], 2,2, model.nhid)

print "average pooling 3x3"
assert average_pool(3).shape == (X.shape[0], 3,3, model.nhid)
