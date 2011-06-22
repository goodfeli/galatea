pca_dim = 100
job_name = 'cfa'

from pylearn2.datasets.mnist import MNIST
from pylearn2.pca import CovEigPCA
import theano.tensor as T
from theano import function
from models import expand
import numpy as N
from scipy.linalg import eigh
from pylearn2.utils import serial
import time
import SkyNet

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


SkyNet.set_job_name(job_name)
components = SkyNet.get_dir_path('components')


serial.save(components+'/pca_model.pkl',pca_model)
serial.save(components+'/whitener.pkl',whitener)
serial.save(components+'/num_examples.pkl',num_examples)
