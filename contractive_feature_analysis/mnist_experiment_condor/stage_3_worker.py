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
import sys 

job_name = sys.argv[1]
idx = int(sys.argv[2])

print 'Loading MNIST train set'
t1 = time.time()
X = MNIST(which_set = 'train').get_design_matrix()
t2 = time.time()
print (t2-t1),' seconds'

print 'HACK: truncating data to 6000 entries'
X = X[0:6000,:]

num_examples, input_dim = X.shape


SkyNet.set_job_name(job_name)
components = SkyNet.get_dir_path('components')

check_num_examples = serial.load(components+'/num_examples.pkl')
assert num_examples == check_num_examples

chunk_size = serial.load(components+'/chunk_size.pkl')
batch_size = serial.load(components+'/batch_size.pkl')

#Restrict X to just the examples we're supposed to run on
X = X[idx:idx+batch_size,:]

pca_model = serial.load(components+'/pca_model.pkl')

print 'Compiling theano PCA function'
t1 = time.time()
pca_input = T.matrix()
pca_output = pca_model(pca_input)
pca_func = function([pca_input],pca_output)
t2 = time.time()
print (t2-t1),' seconds'

print 'Running PCA'
t1 = time.time()
g0full = pca_func(X)
del X
t2 = time.time()
print (t2-t1),' seconds'

P  = pca_model.get_weights()




G = N.zeros((expanded_dim,expanded_dim))
Z = whitener.get_weights()
batch_size = 50

G1 = N.zeros((batch_size,expanded_dim,input_dim))


print 'Computing instability matrix'
for b in xrange(0,chunk_size, batch_size):
    t1 = time.time()
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

    t2 = time.time()
    print (t2-t1),' seconds'

G /= float(chunk_size)

instability_matrices = SkyNet.get_dir_path('instability_matrices')

N.save('instability_matrix_%d.npy' % idx, G)
