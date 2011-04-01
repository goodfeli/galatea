import zipfile
from tempfile import TemporaryFile
import gzip
from pylearn.datasets import utlc
import numpy as N
import scipy
import scipy.sparse
import scipy.linalg
import theano
L = scipy.linalg
import serialutil

bandwidth = 1000.0
k = 50
whiten = True #not the same as whiten in kmeans.py! whiten here controls whether whitening is done before the spectral embedding
tsne = True


def sigmoid(X):
    return 1./(1.+N.exp(-X))
""

def run_network(X):
    return sigmoid(X.dot(W)+b)
""

if tsne:
    X =serialutil.load('../tsne/tsne_terry_30.pkl')
    assert not N.any(N.isnan(X))
    assert not N.any(N.isinf(X))
    valid_start = 4000
    test_start = valid_start + 4096
    #train = X[0:valid_start, :]
    valid = X[valid_start:test_start, :]
    test =  X[test_start:,:]
else:

    W = N.load('./yann_terry/W0.npy')
    b = N.load('./yann_terry/b0.npy')

    print "loading data"
    #train = scipy.sparse.csr_matrix(N.load(gzip.open("/data/lisa/data/UTLC/sparse/terry_train.npy.gz")), dtype=theano.config.floatX)
    #train = train[0:4000,:]
    valid = scipy.sparse.csr_matrix(N.load(gzip.open("/data/lisa/data/UTLC/sparse/terry_valid.npy.gz")), dtype=theano.config.floatX)[1:]
    test = scipy.sparse.csr_matrix(N.load(gzip.open("/data/lisa/data/UTLC/sparse/terry_test.npy.gz")), dtype=theano.config.floatX)[1:]



    print "preprocessing data"
    #train.data = N.sign(train.data)
    valid.data = N.sign(valid.data)
    test.data = N.sign(test.data)

    n2, h =  W.shape

    print "extracting features"

    #train = run_network(train)
    valid = run_network(valid)
    test  = run_network(test)

    #train = train[0:4000,:]

#print 'concatenating datasets'
#X = N.concatenate((train,valid,test),axis=0)

X = valid


if whiten:
    print 'whitening'
    m = X.shape[0]

    print '\tsubtracting mean'
    whitening_mean = X.mean(axis=0)
    X -= whitening_mean

    print '\tfiltering'
    std = X.std(axis=0)
    std[ std < 0.0 ] = 0.0
    filter = N.nonzero(std == 0.0)
    X = X[:, std != 0.0 ]
    n = X.shape[1]

    print '\t\t'+str(n)+' features survived'

    print '\tcomputing covariance'
    cov = N.dot(X.T,X)

    for i in xrange(n):
        if cov[i,i] < 1e-2:
            cov[i,i] = 1e-2

    print '\teigendecomposition'
    vals, vecs = L.eigh(cov)

    if N.any(N.isnan(vals)) or N.any(N.isnan(vecs)):
        raise Exception('nan in eigenvalue solution')

    print '\tmaking new matrix'
    D = N.zeros((n,n))

    for i in xrange(n):
        D[i,i] = 1.0/N.sqrt(1e-10+max(0,vals[i]))

    if N.any(N.isnan(D)):
        raise Exception('nan in D')

    W = N.sqrt(m-1) * N.dot(vecs , N.dot( D , vecs.T))

    print '\tprocessing data'
    X = N.dot(X,W)
#

#print "HACK, PROJECTING TO UNIT SPHERE AFTER WHITENING"
#X = (X.T / N.sqrt(N.square(X).sum(axis=1) )).T




print 'computing similarity matrix'
Z = N.zeros( (X.shape[0],X.shape[0]), X.dtype)
for i in xrange(Z.shape[0]):
    vec = X[i:,:] - X[i,:]
    vec = -N.square(vec).sum(axis=1)
    vec = N.exp( vec / bandwidth)

    Z[i:,i] = vec
    Z[i,i:] = vec
assert not N.any(N.isnan(Z))
assert not N.any(N.isinf(Z))


print 'normalizing similarity matrix'
S = Z.sum(axis=1)
print 'Z: '
print Z
print 'min S: '+str(S.min())
print 'max S: '+str(S.max())
Z /= N.sqrt(1e-10+N.outer(S,S))



print 'computing eigenvectors'

vals, vecs = scipy.linalg.eigh(Z, eigvals=(Z.shape[0]-k,Z.shape[0]-1) )

assert vecs.shape[0] == X.shape[0]
assert vecs.shape[1] == k



valid_start = 4000
test_start = 4000 + valid.shape[0]

#train = vecs[0:valid_start,:]
#valid = vecs[valid_start:test_start,:]
#test = vecs[test_start:,:]

valid = vecs
test = vecs
print "WARNING: just did dummy features for test"


print 'saving'

tsne_str = ""
if tsne:
    tsne_str = "tsne_"


#Save numpy format for subsequent k-means experiment
#N.save('terry_'+tsne_str+'spectral_bandwidth_'+str(bandwidth)+'k_'+str(k)+'train', train)
N.save('terry_'+tsne_str+'spectral_bandwidth_'+str(bandwidth)+'k_'+str(k)+'valid', valid)
N.save('terry_'+tsne_str+'spectral_bandwidth_'+str(bandwidth)+'k_'+str(k)+'test', test)

#Save zip format for direct upload of embedding to server
valid_file = TemporaryFile()
test_file = TemporaryFile()

N.savetxt(valid_file, valid, fmt="%.3f")
N.savetxt(test_file, test, fmt="%.3f")

valid_file.seek(0)
test_file.seek(0)


submission = zipfile.ZipFile("terry_"+tsne_str+"spectral_bandwidth"+str(bandwidth)+"k_"+str(k)+".zip" , "w", compression=zipfile.ZIP_DEFLATED)

submission.writestr("terry_sda_valid.prepro", valid_file.read())
submission.writestr("terry_sda_final.prepro", test_file.read())

submission.close()
valid_file.close()
test_file.close()



