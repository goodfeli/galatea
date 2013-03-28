import sys
import SkyNet
from pylearn2.utils import serial
import theano.tensor as T
from theano import function
import numpy as N
from theano import config
floatX = config.floatX

job_name = sys.argv[1]
SkyNet.set_job_name(job_name)

components = SkyNet.get_dir_path('components')

data = serial.load(components+'/dataset.pkl')
X = N.cast[floatX](data.get_design_matrix())

m,n = X.shape

pca_model = serial.load(components+'/pca_model.pkl')
whitener = serial.load(components+'/whitener.pkl')
W = serial.load(components+'/W.pkl')
fe = serial.load(components+'/fe.pkl')
fe.redo_theano()

ipt = T.matrix()
pca = pca_model(ipt)
pca_func  = function([ipt],pca)
white = whitener(ipt)
whiten_func = function([ipt],white)

def process(D):
    rval = pca_func(D)
    #rval = D
    rval = fe.extract(rval)
    rval = whiten_func(rval)
    rval = N.dot(rval,W)
    return rval

nn =  whitener.get_weights().shape[1]
#nn = W.shape[1]

mu = N.zeros(nn)

batch_size = 100
assert m % batch_size == 0
num_batches = m / batch_size

"""for i in xrange(0,m,batch_size):
    print i
    mu += process(X[i:i+batch_size,:]).sum(axis=0)

mu /= float(m)"""

g1= process(X)


mu = g1.mean(axis=0)

del g1


print 'mean: ',(mu.min(),mu.mean(),mu.max())

cov = N.zeros((nn,nn))

for i in xrange(0,m,batch_size):
    print i
    g = process(X[i:i+batch_size,:])
    g -= mu
    cov += N.dot(g.T,g)

cov /= float(m-1)

var = cov.diagonal()

print 'var: ',(var.min(),var.mean(),var.max())

for i in xrange(nn):
    cov[i,i] = 0.

print 'cov: ',(cov.min(),N.abs(cov).sum()/float(cov.shape[0] ** 2 - cov.shape[0]),cov.max())
