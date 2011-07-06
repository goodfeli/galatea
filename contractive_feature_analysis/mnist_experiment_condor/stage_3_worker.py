#!/bin/env python
exp_name = 'cfa'

from pylearn2.datasets.mnist import MNIST
import theano.tensor as T
from theano import function
from models import expand
import numpy as N
from pylearn2.utils import serial
import time
import sys
import SkyNet

job_name = sys.argv[1]
idx = int(sys.argv[2])

SkyNet.set_job_name(job_name)
components = SkyNet.get_dir_path('components')
whitener = serial.load(components+'/whitener.pkl')


print 'Loading MNIST train set'
t1 = time.time()
X = serial.load(components+'/dataset.pkl').get_design_matrix()
t2 = time.time()
print (t2-t1),' seconds'

num_examples, input_dim = X.shape

check_num_examples = serial.load(components+'/num_examples.pkl')
assert num_examples == check_num_examples

chunk_size = serial.load(components+'/chunk_size.pkl')
batch_size = serial.load(components+'/batch_size.pkl')
expanded_dim = serial.load(components+'/expanded_dim.pkl')

#Restrict X to just the examples we're supposed to run on
X = X[idx:idx+chunk_size,:]

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

G1 = N.zeros((batch_size,expanded_dim,input_dim))

assert chunk_size % batch_size == 0

print 'Computing instability matrix'
for b in xrange(0,chunk_size, batch_size):
    t1 = time.time()
    g0 = g0full[b:b+batch_size,:]

    #print 'Computing Jacobian of basis expansion'
    J = expand.jacobian_of_expand(g0)

    #print 'Computing Jacobian of basis expansion composed with PCA'
    for i in xrange(batch_size):
        #print J.shape
        print i,G1[i,:,:].shape, J[i,:,:].shape, P.shape
        G1[i,:,:] = N.dot(J[i,:,:],P.T)
    del J

    #print 'Computing final (post-whitening) Jacobian'
    G3 = G1

    for i in xrange(batch_size):
        #verified that this is Z.T and not Z by running a test with one whitened component dropped
        G3[i,:,:]  = N.dot(Z.T, G3[i,:,:])

    #print 'Computing instability matrix'
    for i in xrange(batch_size):
        G += N.dot(G3[i,:,:],G3[i,:,:].T) / float(batch_size)

    t2 = time.time()
    print (t2-t1),' seconds'

G /= float(chunk_size)

instability_matrices = SkyNet.get_dir_path('instability_matrices')

N.save(instability_matrices+'/instability_matrix_%d.npy' % idx, G)
