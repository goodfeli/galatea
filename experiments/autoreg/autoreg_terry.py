import zipfile
from tempfile import TemporaryFile
import gzip
from pylearn.datasets import utlc
import numpy as N
import scipy
import scipy.sparse
import scipy.linalg
import theano


reg_coeff = 1e6

def sigmoid(X):
    return 1./(1.+N.exp(-X))
""

def run_network(X):
    return sigmoid(X.dot(W)+b)
""

def compute_reg(X):

    print '\tmaking regularized covariance matrix'
    z = N.dot(X.T,X)+reg_coeff*N.identity(X.shape[1])
    print '\tinverting regularized covariance matrix'
    y = scipy.linalg.solve(z,N.identity(X.shape[1]), sym_pos = True)
    print '\ttaking square root of matrix inverse'
    return scipy.linalg.sqrtm(y)

W = N.load('./yann_terry/W0.npy')
b = N.load('./yann_terry/b0.npy')

print "loading data"
#train = scipy.sparse.csr_matrix(N.load(gzip.open("/data/lisa/data/UTLC/sparse/terry_train.npy.gz")), dtype=theano.config.floatX)
valid = scipy.sparse.csr_matrix(N.load(gzip.open("/data/lisa/data/UTLC/sparse/terry_valid.npy.gz")), dtype=theano.config.floatX)[1:]
test = scipy.sparse.csr_matrix(N.load(gzip.open("/data/lisa/data/UTLC/sparse/terry_test.npy.gz")), dtype=theano.config.floatX)[1:]



print "preprocessing data"
#train.data = N.sign(train.data)
valid.data = N.sign(valid.data)
test.data = N.sign(test.data)

n2, h =  W.shape

print "extracting features"

valid = run_network(valid)
test  = run_network(test)

print 'computing regularization'

valid_reg = compute_reg(valid)
test_reg = compute_reg(test)

print 'applying regularization'

valid = N.dot(valid, valid_reg)
test = N.dot(test, test_reg)


print 'saving'

valid_file = TemporaryFile()
test_file = TemporaryFile()

N.savetxt(valid_file, valid, fmt="%.3f")
N.savetxt(test_file, test, fmt="%.3f")

valid_file.seek(0)
test_file.seek(0)

submission = zipfile.ZipFile("terry_autoreg_"+str(reg_coeff)+".zip" , "w", compression=zipfile.ZIP_DEFLATED)

submission.writestr("terry_sda_valid.prepro", valid_file.read())
submission.writestr("terry_sda_final.prepro", test_file.read())

submission.close()
valid_file.close()
test_file.close()



