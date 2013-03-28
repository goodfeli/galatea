pca_dim = 100
exp_name = 'cfa'

from pylearn2.datasets.mnist import MNIST
from pylearn2.pca import CovEigPCA
import theano.tensor as T
from theano import function
from models import expand
import numpy as N
from scipy.linalg import eigh
from pylearn2.utils import serial
import time

print 'Loading MNIST train set'
t1 = time.time()
X = MNIST(which_set = 'train').get_design_matrix()
t2 = time.time()
print (t2-t1),' seconds'

print 'HACK: truncating data to 6000 entries'
X = X[0:6000,:]

num_examples, input_dim = X.shape

print 'Training PCA with %d dimensions' % pca_dim
t1 = time.time()
pca_model = CovEigPCA(num_components = pca_dim)
pca_model.train(X)
t2 = time.time()
print (t2-t1),' seconds'

print 'Compiling theano PCA function'
t1 = time.time()
pca_input = T.matrix()
pca_output = pca_model(pca_input)
pca_func = function([pca_input],pca_output)
t2 = time.time()
print (t2-t1),' seconds'

print 'Running PCA'
t1 = time.time()
g0 = pca_func(X)
del X
t2 = time.time()
print (t2-t1),' seconds'

P  = pca_model.get_weights()


print 'Computing basis expansion'
t1 = time.time()
g1 = expand.expand(g0)
expanded_dim = g1.shape[1]
t2 = time.time()
print (t2-t1),' seconds'

print 'Whitening expanded data'
t1 = time.time()
whitener = CovEigPCA(num_components = expanded_dim, whiten=True)
whitener.train(g1)
t2 = time.time()
print (t2-t1),' seconds'

del g1


G = N.zeros((expanded_dim,expanded_dim))
Z = whitener.get_weights()
batch_size = 50

G1 = N.zeros((batch_size,expanded_dim,input_dim))

g0full = g0
print 'Computing instability matrix'
t1 = time.time()
for b in xrange(0,num_examples,batch_size):
    print '\texample ',b

    t1a = time.time()
    g0 = g0full[b:b+batch_size,:]

    #print 'Computing Jacobian of basis expansion'
    J = expand.jacobian_of_expand(g0)

    #print 'Computing Jacobian of basis expansion composed with PCA'
    for i in xrange(batch_size):
        #print G1[i,:,:].shape, J[i,:,:].shape, P.shape
        G1[i,:,:] = N.dot(J[i,:,:],P.T)
    del J

    #print 'Computing final (post-whitening) Jacobian'
    G3 = G1
    del G1
    for i in xrange(batch_size):
        #TODO: should this be Z.T ?
        G3[i,:,:]  = N.dot(Z, G3[i,:,:])

    #print 'Computing instability matrix'
    for i in xrange(batch_size):
        G += N.dot(G3[i,:,:],G3[i,:,:].T) / float(batch_size)

    t1b = time.time()
    print '\t',(t1b-t1a),' seconds'
t2 = time.time()
print (t2-t1),' seconds'

print 'Finding eigenvectors'
t1 = time.time()
v, W = eigh(G)
t2 = time.time()
print (t2-t1),' seconds'

results = {}
results['v'] = v
results['W'] = W
results['whitener'] = whitener
results['pca_model'] = pca_model

serial.save('mnist_results.pkl',results)
