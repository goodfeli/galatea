pca_dim = 16
job_name = 'cfa_test'

from pylearn2.datasets.mnist import MNIST
from pylearn2.pca import CovEigPCA
import theano.tensor as T
from theano import function
from models import expand
from pylearn2.utils import serial
import time
import SkyNet
import gc
import numpy as N

print 'Loading MNIST train set'
t1 = time.time()
X = MNIST(which_set = 'train').get_design_matrix()
t2 = time.time()
print (t2-t1),' seconds'


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

SkyNet.set_job_name(job_name)
components = SkyNet.get_dir_path('components')
serial.save(components+'/pca_model.pkl',pca_model)

del pca_model
del pca_output
del pca_func
gc.collect()

print 'Computing basis expansion'
t1 = time.time()
g1 = expand.expand(g0)


expanded_dim = g1.shape[1]
t2 = time.time()
print (t2-t1),' seconds'

del g0
gc.collect()

print 'Whitening expanded data'
t1 = time.time()
whitener = CovEigPCA(cov_batch_size = 50, num_components = expanded_dim, whiten=True)
print g1.shape
whitener.train(g1)
t2 = time.time()
print (t2-t1),' seconds'


serial.save(components+'/whitener.pkl',whitener)
serial.save(components+'/num_examples.pkl',num_examples)
serial.save(components+'/expanded_dim.pkl',expanded_dim)


#checks
out = whitener(pca_input)
out_func = function([pca_input],out)
g1 = out_func(N.cast['float32'](g1))

mu = g1.mean(axis=0)
print (mu.min(),mu.max())
std = g1.mean(axis=0)
print (std.min(),std.max())
