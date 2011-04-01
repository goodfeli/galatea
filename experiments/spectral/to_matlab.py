import zipfile
from tempfile import TemporaryFile
import gzip
import numpy as N
import scipy
import theano
from scipy import io
L = scipy.linalg

src= 'yann'
valid_only = False

def sigmoid(X):
    return 1./(1.+N.exp(-X))
""

def run_network(X):
    return sigmoid(X.dot(W)+b)
""

if src == 'xavier':
    valid = N.loadtxt('terry_dl15_valid.prepro')
    valid_only = True
else:

    W = N.load('./yann_terry/W0.npy')
    b = N.load('./yann_terry/b0.npy')

    print "loading data"
    train = scipy.sparse.csr_matrix(N.load(gzip.open("/data/lisa/data/UTLC/sparse/terry_train.npy.gz")), dtype=theano.config.floatX)
    train = train[0:4000,:]
    valid = scipy.sparse.csr_matrix(N.load(gzip.open("/data/lisa/data/UTLC/sparse/terry_valid.npy.gz")), dtype=theano.config.floatX)[1:]
    test = scipy.sparse.csr_matrix(N.load(gzip.open("/data/lisa/data/UTLC/sparse/terry_test.npy.gz")), dtype=theano.config.floatX)[1:]



    print "preprocessing data"
    train.data = N.sign(train.data)
    valid.data = N.sign(valid.data)
    test.data = N.sign(test.data)

    n2, h =  W.shape

    print "extracting features"

    train = run_network(train)
    valid = run_network(valid)
    test  = run_network(test)


print 'concatenating datasets'
if valid_only:
    X = valid
else:
    X = N.concatenate((train,valid,test),axis=0)


io.savemat('X',{ 'X' : X })


