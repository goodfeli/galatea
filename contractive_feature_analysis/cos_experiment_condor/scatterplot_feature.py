import sys
import SkyNet
from pylearn2.utils import serial
import theano.tensor as T
from theano import function
import numpy as N
from theano import config
floatX = config.floatX
from matplotlib import pyplot as plt

job_name = sys.argv[1]
idx = int(sys.argv[2])
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
    rval = N.dot(rval,W[:,idx])
    return rval

nn =  whitener.get_weights().shape[1]
#nn = W.shape[1]

mu = N.zeros(nn)

batch_size = 100
assert m % batch_size == 0
num_batches = m / batch_size

g1 = N.zeros(m)

for i in xrange(0,m,batch_size):
    g1[i:i+batch_size] = process(X[i:i+batch_size,:])

plt.scatter(X[:,0],X[:,1], c = g1)
plt.show()
