from pylearn2.datasets.cifar10 import CIFAR10
import numpy as np
from galatea.cond.neighbs import cifar10neighbs


m = 10
r = 6
c = 6
dataset = CIFAR10(which_set = 'train', one_hot = True, gcn = 55.)

ten4 = dataset.get_batch_topo(m)

from pylearn2.utils import sharedX
ten4th = sharedX(ten4)

X = cifar10neighbs(ten4, (r,c))

from theano import function

X = function([],X)()
print X.shape

from pylearn2.gui.patch_viewer import make_viewer
from pylearn2.utils.image import show
stride = (32-r+1)*(32-c+1)

for i in xrange(m):
    ten4v =ten4[i,:,:,:]
    ten4v -= ten4v.min()
    ten4v /= ten4v.max()
    show(ten4v)
    patch_viewer = make_viewer(X[i*stride:(i+1)*stride], is_color= True)
    patch_viewer.show()

    print 'waiting...'
    x = raw_input()
