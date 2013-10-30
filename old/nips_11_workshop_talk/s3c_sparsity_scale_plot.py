from pylearn2.utils import serial
import sys
from pylearn2.config.yaml_parse import load
import theano.tensor as T
from theano import function
import numpy as np
import matplotlib.pyplot as plt
from theano import config
#float32 seems to break David's code
config.floatX = 'float32'

print 'loading model'
model = serial.load(sys.argv[1])
model.make_pseudoparams()
model.set_dtype(config.floatX)

print 'loading dataset'
dataset = load(model.dataset_yaml_src)

print 'compiling function'
V = T.fmatrix()
obs = model.e_step.variational_inference(V)
H = obs['H_hat']
S = obs['S_hat']
HS = abs(H*S)

f = function([V], HS)

print 'running inference'
batch_size = 5000

X = dataset.get_batch_design(batch_size)
HS = f(X)

print 'processing results'
act_prob = (HS > .01).mean(axis=0)

act_mean = np.zeros(act_prob.shape)
for i in xrange(act_mean.shape[0]):
    s = HS[:,i]
    s = s[s > .01]
    act_mean[i] = s.mean()

print 'drawing plot'
plt.hexbin(act_prob, act_mean)
plt.show()


"""
print 'making SC'
from sklearn.decomposition import MiniBatchDictionaryLearning
m = MiniBatchDictionaryLearning(n_atoms = model.nhid, fit_algorithm='lars', transform_algorithm = 'lasso_lars',
        dict_init = model.W.get_value().T)
m.components_ = model.W.get_value().T

print 'running SC inference'
HS = m.transform(X)
"""

"""
rng = np.random.RandomState([1,2,3])
from pylearn2.utils import sharedX

HS = sharedX(np.dot(X,model.W.get_value()) * model.nvis / model.nhid)

obj = T.mean(T.sqr(T.dot(HS,model.W.T)-V)) + T.mean(abs(HS))

g = T.grad(obj,HS)

learning_rate = 10.

f = function([V], obj, updates = { HS : HS - g * learning_rate } )

n_iter = 1000

for i in xrange(n_iter):
    print f(X)

HS = HS.get_value()
"""

from sklearn.decomposition import sparse_encode
print 'running SC'
HS = sparse_encode( model.W.get_value(), X.T, alpha = 1./5000., algorithm='lasso_cd').T


"""
print 'running featuresign'
from pylearn2.optimization.feature_sign import feature_sign_search
HS = feature_sign_search(model.W.get_value(), X, 1.)
"""

HS = np.abs(HS)

print HS.shape
print (HS.min(),HS.max(),HS.mean())

if np.any(np.isnan(HS)):
    print 'has nans'

if np.any(np.isinf(HS)):
    print 'has infs'

print 'processing results'
act_prob = (HS > .01).mean(axis=0)

act_mean = np.zeros(act_prob.shape)
for i in xrange(act_mean.shape[0]):
    s = HS[:,i]
    s = s[s > .01]
    act_mean[i] = s.mean()

#print act_prob
#print act_mean

print 'drawing plot'
plt.hexbin(act_prob, act_mean)
plt.show()
