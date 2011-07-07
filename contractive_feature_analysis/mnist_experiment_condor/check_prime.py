import sys
import SkyNet
from pylearn2.utils import serial
import theano.tensor as T
from theano import function
from models import expand
import numpy as N

job_name = sys.argv[1]
SkyNet.set_job_name(job_name)

components = SkyNet.get_dir_path('components')

data = serial.load(components+'/dataset.pkl')
X = N.cast['float32'](data.get_design_matrix())

m,n = X.shape

pca_model = serial.load(components+'/pca_model.pkl')
whitener = serial.load(components+'/whitener.pkl')
W = serial.load(components+'/W.pkl')

print N.square(W).sum(axis=0)
quit()


ipt = T.matrix()
pca = pca_model(ipt)
pca_func  = function([ipt],pca)
white = whitener(ipt)
whiten_func = function([ipt],white)

def process(D):
    rval = pca_func(D)
    rval = N.cast['float32'](expand.expand(rval))
    rval = whiten_func(rval)
    rval = N.dot(rval,W)
    return rval

nn = W.shape[1]

mu = N.zeros(nn)

batch_size = 100
assert m % batch_size == 0
num_batches = m / batch_size

for i in xrange(0,m,batch_size):
    print i
    mu += process(X[i:i+batch_size,:]).sum(axis=0)

mu /= float(m)

print 'mean: ',(mu.min(),mu.mean(),mu.max())

cov = N.zeros((nn,nn))

for i in xrange(0,m,batch_size):
    print i
    g = process(X[i:i+batch_size,:])
    g -= mu
    cov += N.dot(g.T,g)

cov /= (m-1)

var = cov.diagonal()

print 'var: ',(var.min(),var.mean(),var.max())

for i in xrange(n):
    cov[i,i] = 0.

print 'cov: ',(cov.min(),cov.max())
