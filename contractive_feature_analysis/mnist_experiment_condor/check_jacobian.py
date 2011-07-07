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

ipt = T.matrix()
pca = pca_model(ipt)
pca_func  = function([ipt],pca)
white = whitener(ipt)
whiten_func = function([ipt],white)

def subprocess(D):
    rval = pca_func(D)
    #rval = D
    rval = N.cast['float32'](expand.expand(rval))
    rval = whiten_func(rval)
    rval = N.dot(rval,W)
    return rval

rng = N.random.RandomState([1,2,3])
eps = 1e-6

def process(D):
    g0 = subprocess(D)
    D += rng.uniform(-eps,eps,D.shape)
    g1 = subprocess(D)
    diff = g0 - g1
    dir_grad = diff / eps
    return N.square(dir_grad)

nn =  whitener.get_weights().shape[1]
#nn = W.shape[1]

mu = N.zeros(nn)

batch_size = 100
assert m % batch_size == 0
num_batches = m / batch_size

for i in xrange(0,m,batch_size):
    print i
    mu += process(X[i:i+batch_size,:]).sum(axis=0)

mu /= float(m)

print mu

