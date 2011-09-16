from pylearn2.datasets.cifar10 import CIFAR10
from theano import config
from theano import tensor as T
from theano.sandbox.neighbours import images2neibs
from theano import function

print 'loading dataset'
dataset = CIFAR10(which_set = "train")

batch_start = 0
batch_size = 100

X = dataset.get_design_matrix()[batch_start:batch_start + batch_size,:]
topo_val = dataset.get_topological_view(X)


print 'writing theano program'

topo_var = T.TensorType(dtype = config.floatX, broadcastable = (False, False, False, False) )

topo_reformatted = topo_var.dimshuffle(0,3,1,2)

neibs = images2neibs(ten4 = topo_reformatted, neib_shape = (8,8))

print 'compiling theano function'

f = function([topo_var], neibs)

print 'computing neighborhoods'
neibs = f()

print 'neibs shape: ',neibs.shape


