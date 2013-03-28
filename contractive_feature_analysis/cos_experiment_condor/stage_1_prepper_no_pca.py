job_name = 'cfa_cos_tanh'

import numpy as N
from pylearn2.pca import CovEigPCA
from pylearn2.utils import serial
import time
import SkyNet
import gc
from pylearn2.datasets.cos_dataset import CosDataset
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from feature_extractor import TanhFeatureExtractor
from theano import config
floatX = config.floatX

print 'making dataset'
t1 = time.time()
data = CosDataset()
X = N.cast[floatX](data.get_batch_design(50000))
data = DenseDesignMatrix(X)
t2 = time.time()
print (t2-t1),' seconds'


num_examples, input_dim = X.shape

fe = TanhFeatureExtractor.make_from_examples(X[0:72,:], -.99, .99, directed = False)



pca_dim = X.shape[1]
print 'Training PCA with %d dimensions' % pca_dim
t1 = time.time()
pca_model = CovEigPCA(num_components = pca_dim)
pca_model.train(X)
pca_model.W.set_value(N.cast['float32'](N.identity(X.shape[1])))
assert pca_model.get_weights().shape[1] == pca_dim
pca_model.mean.set_value(N.cast['float32'](N.zeros(X.shape[1])))
t2 = time.time()
print (t2-t1),' seconds'


SkyNet.set_job_name(job_name)
components = SkyNet.get_dir_path('components')
serial.save(components+'/fe.pkl',fe)
serial.save(components+'/pca_model.pkl',pca_model)
serial.save(components+'/dataset.pkl',data)

g0 = X

print 'Computing basis expansion'
t1 = time.time()
g1 = fe.extract(g0)


expanded_dim = g1.shape[1]
t2 = time.time()
print (t2-t1),' seconds'

del g0
gc.collect()

print 'Whitening expanded data'
t1 = time.time()
whitener = CovEigPCA(cov_batch_size = 50, num_components = expanded_dim, whiten=True)
print 'white components: '+str(expanded_dim)
print 'expanded dataset shape: '+str(g1.shape)
whitener.train(g1)
t2 = time.time()
print (t2-t1),' seconds'

Z = whitener.get_weights()

serial.save(components+'/whitener.pkl',whitener)
serial.save(components+'/num_examples.pkl',num_examples)
serial.save(components+'/expanded_dim.pkl',expanded_dim)


print 'done, checking result'

#checks
from theano import function
import theano.tensor as T
pca_input = T.matrix()
assert pca_input.dtype == floatX

del whitener
whitener = serial.load(components+'/whitener.pkl')

out = whitener(pca_input)
assert out.dtype == floatX
out_func = function([pca_input],out)
test = out_func((g1))

#print g1[0:5,0:5]



"""g1 -= whitener.mean.get_value()
print 'after manual mean subtract, mean is'
mu = g1.mean(axis=0)
print (mu.min(), mu.max())
g1 = N.dot(g1,whitener.get_weights())
print 'after manual whitening, mean is '
"""

mu = test.mean(axis=0)
print (mu.min(), mu.max())

print 'standard deviation'
std = test.std(axis=0)
print (std.min(), std.max())

whitener.W.set_value(whitener.W.get_value() / std )

test = out_func(g1)


mu = test.mean(axis=0)
print (mu.min(), mu.max())

print 'standard deviation'
std = test.std(axis=0)
print (std.min(), std.max())

serial.save(components+'/whitener.pkl',whitener)
