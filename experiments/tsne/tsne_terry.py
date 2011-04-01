import gzip
from Tsne import calc_tsne
from pylearn.datasets import utlc
import serialutil 
import numpy as N
import scipy
import scipy.sparse
import theano

def sigmoid(X):
    return 1./(1.+N.exp(-X))
""

def run_network(X):
    return sigmoid(X.dot(W)+b)
""

W = N.load('./yann_terry/W0.npy')
b = N.load('./yann_terry/b0.npy')

print "loading data"
train = scipy.sparse.csr_matrix(N.load(gzip.open("/data/lisa/data/UTLC/sparse/terry_train.npy.gz")), dtype=theano.config.floatX)
valid = scipy.sparse.csr_matrix(N.load(gzip.open("/data/lisa/data/UTLC/sparse/terry_valid.npy.gz")), dtype=theano.config.floatX)[1:]
test = scipy.sparse.csr_matrix(N.load(gzip.open("/data/lisa/data/UTLC/sparse/terry_test.npy.gz")), dtype=theano.config.floatX)[1:]



print "preprocessing data"
train.data = N.sign(train.data)
valid.data = N.sign(valid.data)
test.data = N.sign(test.data)

n2, h =  W.shape

print "extracting featues"
train = run_network(train)
valid = run_network(valid)
test  = run_network(test)

train = train[0:4000,:]


print "concatenating features"
X = N.concatenate((train,valid,test),axis=0)

del train
del valid
del test


print "running t-sne"

proj = calc_tsne.calc_tsne(X, NO_DIMS=2)

serialutil.save('tsne_terry.pkl',proj)
