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

print 'loading dataset'
dataset = CIFAR10(which_set = "train")

batch_start = 0
batch_size = 5

X = dataset.get_design_matrix()[batch_start:batch_start + batch_size,:]

pv1 = make_viewer(X, is_color = True)
dataset.set_design_matrix(X)

patchifier = ExtractGridPatches( patch_shape = (8,8), patch_stride = (1,1) )



pipeline = serial.load('${GOODFELI_TMP}/cifar10_preprocessed_pipeline_2M.pkl')

assert isinstance(pipeline.items[0], ExtractPatches)
pipeline.items[0] = patchifier

print 'applying preprocessor'
dataset.apply_preprocessor(pipeline, can_fit = False)


X2 = dataset.get_design_matrix()


import sys
model_path = sys.argv[1]

print 'loading model'
model = serial.load(model_path)

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
depatchifier = ReassembleGridPatches( orig_shape  = (25, 25), patch_shape=(1,1) )
feat_dataset.apply_preprocessor(depatchifier)

print 'making topological view'
topo_feat = feat_dataset.get_topological_view()
assert topo_feat.shape[0] == X.shape[0]

print 'assembling visualizer'
k = 5
pv3 = PatchViewer(grid_shape = (X.shape[0], k), patch_shape=(25,25), is_color= False)
pv4 = PatchViewer(grid_shape = (k, 1), patch_shape = (8,8), is_color = True)

idx = sorted(range(model.nhid), key = lambda l : -topo_feat[:,:,:,l].std() )

W = model.W.get_value()

call weights view and use that that

for j in xrange(k):
    pv4.add_patch(weights at idx[j], weights = True), rescale = True)

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
