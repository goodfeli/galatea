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

print "loading data"
train = N.load('/data/lisa/exp/dauphiya/stackedterry/best_layer1/terry_train.npy')
valid = N.load('/data/lisa/exp/dauphiya/stackedterry/best_layer1/terry_valid.npy')
test = N.load('/data/lisa/exp/dauphiya/stackedterry/best_layer1/terry_test.npy')


train = train[0:4000,:]


print "concatenating features"
X = N.concatenate((train,valid,test),axis=0)

del train
del valid
del test


print "running t-sne"

proj = calc_tsne.calc_tsne(X, NO_DIMS=2)

serialutil.save('tsne_for_yann.pkl',proj)
