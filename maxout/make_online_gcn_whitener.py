from pylearn2.datasets.cifar10 import CIFAR10
import numpy as np

from galatea.maxout import pad

dataset = CIFAR10(which_set='train', axes = ('c', 0, 1, 'b'))

dataset = pad(dataset=dataset, amt=8)

from galatea.maxout import GCN_C01B2

layer = GCN_C01B2(layer_name='unused')

from pylearn2.space import Conv2DSpace

space = Conv2DSpace(shape=[32, 32], num_channels=3, axes = ('c', 0, 1, 'b'))

layer.set_input_space(space)

from pylearn2.utils import function

X = space.make_batch_theano()

gcn = function([X], layer.fprop(X))

from pylearn2.space import VectorSpace
vector_space = VectorSpace(32*32*3)

flatten = function([X], space.format_as(X, vector_space))

mean = np.zeros((32*32*3,), dtype='float32')
cov = np.zeros((32*32*3, 32*32*3), dtype='float32')

dataset.X = dataset.X.astype('float32')

r_ofs = 8
c_ofs = 8


filter_bias = 0.1

for i in xrange(r_ofs):
    print i
    for j in xrange(c_ofs):
        print '\t',j
        for flip in [0, 1]:
            print '\t\t',flip
            X = dataset.get_topological_view(dataset.X)[:, i:i+32, j:j+32, :]
            if flip:
                X = X[:,:,::-1,:]
            X = gcn(X)
            X = flatten(X)

            assert X.dtype in ['float32']
            assert len(X.shape) == 2
            n_samples = X.shape[0]
            mean += np.mean(X, axis=0)
mean /= (r_ofs * c_ofs * 2)


for i in xrange(r_ofs):
    print i
    for j in xrange(c_ofs):
        print '\t',j
        for flip in [0, 1]:
            X = dataset.get_topological_view(dataset.X)[:, i:i+32, j:j+32, :]
            if flip:
                X = X[:,:,::-1,:]
            X = gcn(X)
            X = flatten(X)

            assert X.dtype in ['float32']
            assert len(X.shape) == 2
            n_samples = X.shape[0]
            # Center data
            X -=  mean
            cov += np.dot(X.T, X) /n_samples
cov /= (r_ofs * c_ofs * 2)

from numpy import linalg
eigs, eigv = linalg.eigh(cov)
print 'done with eigh'
assert not np.any(np.isnan(eigs))
assert not np.any(np.isnan(eigv))
P = np.dot(eigv * np.sqrt(1.0 / (eigs + filter_bias)),
                 eigv.T)
assert not np.any(np.isnan(P))

from pylearn2.utils import serial

serial.save('online_whitener.pkl', {'mean': mean, 'P': P } )
