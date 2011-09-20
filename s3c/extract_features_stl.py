import time
import copy
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

config.floatX = 'float32'

batch_size = 5

import sys
model_path = sys.argv[1]

print 'loading model'
model = serial.load(model_path)
model.set_dtype('float32')

print 'loading dataset'
dataset = serial.load('${PYLEARN2_DATA_PATH}/stl10/stl10_32x32/train.pkl')
full_X = dataset.get_design_matrix()
dataset.X = None
dataset.design_loc = None
dataset.compress = False

size = np.sqrt(model.nvis/3)

patchifier = ExtractGridPatches( patch_shape = (size,size), patch_stride = (1,1) )

if size ==6:
    pipeline = serial.load('${PYLEARN2_DATA_PATH}/stl10/stl10_patches/preprocessor.pkl')
else:
    print size
    assert False

assert isinstance(pipeline.items[0], ExtractPatches)
pipeline.items[0] = patchifier


print 'defining features'
V = T.matrix()
model.make_Bwp()
d = model.e_step.mean_field(V = V)

H = d['H']
Mu1 = d['Mu1']

assert H.dtype == 'float32'
assert Mu1.dtype == 'float32'

feat = H * Mu1

assert feat.dtype == 'float32'
print 'compiling theano function'
f = function([V],feat)

topo_feat_var = T.TensorType(broadcastable = (False,False,False,False), dtype='float32')()
region_features = function([topo_feat_var],
        topo_feat_var.mean(axis=(1,2)) )

def average_pool( stride ):
    def point( p ):
        return p * ns / stride

    rval = np.zeros( (topo_feat.shape[0], stride, stride, topo_feat.shape[3] ) , dtype = 'float32')

    for i in xrange(stride):
        for j in xrange(stride):
            rval[:,i,j,:] = region_features( topo_feat[:,point(i):point(i+1), point(j):point(j+1),:] )

    return rval
assert full_X.shape[0] == 5000

#out4 = np.zeros((5000,2,2,model.nhid),dtype='float32')
#out9 = np.zeros((5000,3,3,model.nhid),dtype='float32')
out16 = np.zeros((5000,4,4,model.nhid),dtype='float32')

fd = DenseDesignMatrix(X = np.zeros((1,1),dtype='float32'), view_converter = DefaultViewConverter([1, 1, model.nhid] ) )

ns = 32 - size + 1
depatchifier = ReassembleGridPatches( orig_shape  = (ns, ns), patch_shape=(1,1) )

for i in xrange(0,5000-batch_size+1,batch_size):
    print i
    t1 = time.time()

    d = copy.copy(dataset)
    d.set_design_matrix(full_X[i:i+batch_size,:])

    t2 = time.time()

    #print '\tapplying preprocessor'
    d.apply_preprocessor(pipeline, can_fit = False)
    X2 = d.get_design_matrix()

    t3 = time.time()

    #print '\trunning theano function'
    feat = f(X2)

    t4 = time.time()

    assert feat.dtype == 'float32'

    feat_dataset = copy.copy(fd)
    feat_dataset.set_design_matrix(feat)

    #print '\treassembling features'
    feat_dataset.apply_preprocessor(depatchifier)

    #print '\tmaking topological view'
    topo_feat = feat_dataset.get_topological_view()
    assert topo_feat.shape[0] == batch_size

    t5 = time.time()

    #print '\taverage pooling 2x2'
    #out4[i:i+batch_size,...] =  average_pool(2)
    #out9[i:i+batch_size,...] =  average_pool(3)
    out16[i:i+batch_size,...] =  average_pool(4)

    t6 = time.time()

    print (t6-t1, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5)

#serial.save('out4.pkl',out4)
#serial.save('out9.pkl',out9)
np.save('out16.npy',out16)
