job_name = 'cfa_olshausen'

import numpy as N
from pylearn2.pca import CovEigPCA
from models import expand
from pylearn2.utils import serial
import time
import SkyNet
import gc

print 'Loading MNIST train set'
t1 = time.time()
data =  serial.load('olshausen.pkl')
X = data.get_design_matrix()
X = X[0:20000,:]
t2 = time.time()
print (t2-t1),' seconds'



num_examples, input_dim = X.shape

pca_dim = X.shape[1]
print 'Training PCA with %d dimensions' % pca_dim
t1 = time.time()
pca_model = CovEigPCA(num_components = pca_dim)
pca_model.train(X)
pca_model.W.set_value(N.cast['float32'](N.identity(X.shape[1])))
pca_model.mean.set_value(N.cast['float32'](N.zeros(X.shape[1])))
t2 = time.time()
print (t2-t1),' seconds'


SkyNet.set_job_name(job_name)
components = SkyNet.get_dir_path('components')
serial.save(components+'/pca_model.pkl',pca_model)
serial.save(components+'/dataset.pkl',data)

g0 = X

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

print 'done, checking result'

#checks
from theano import function
import theano.tensor as T
pca_input = T.matrix()

del whitener
whitener = serial.load(components+'/whitener.pkl')

out = whitener(pca_input)
out_func = function([pca_input],out)
g1 = out_func(N.cast['float32'](g1))

print g1[0:5,0:5]

mu = g1.mean(axis=0)
print (mu.min(),mu.max())
std = g1.std(axis=0)
print (std.min(),std.max())
