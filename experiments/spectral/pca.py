import zipfile
from tempfile import TemporaryFile
import gzip
import numpy as N
import scipy
import theano
L = scipy.linalg

k = 500
add = 0.0
src= 'harry_best'
whiten = False #not the same as whiten in kmeans.py! whiten here controls whether whitening is done before the spectral embedding
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
elif src == 'harry_best':
    valid = N.loadtxt('harry_best_valid.prepro')
    valid_only = True
elif src == 'yann':

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

    W = N.sqrt(float(m-1)) * N.dot(vecs , N.dot( D , vecs.T))

    print '\tprocessing data'
    X = N.dot(X,W)
#

#print "HACK, PROJECTING TO UNIT SPHERE AFTER WHITENING"
#X = (X.T / N.sqrt(N.square(X).sum(axis=1) )).T

print 'doing pca'
X -= X.mean(axis=0)
X /= X.std(axis=0)
cov = N.dot(X.T,X) / N.sqrt(float(X.shape[0]-1))

for i in xrange(cov.shape[0]):
        X[i,i] += add

vals, vecs = L.eigh(cov, eigvals=(cov.shape[0]-k,cov.shape[0]-1))

vecs *= (vals ** 2.)

X = N.dot(X,vecs)


if valid_only:
        print 'warning, dummy featues for test'
        valid = X
        test = valid
else:
    valid_start = train.shape[0]
    test_start = valid_start + valid.shape[0]

    valid = X[valid_start:test_start,:]
    test = X[test_start:,:]


print 'saving'



#Save zip format for direct upload of embedding to server
valid_file = TemporaryFile()
test_file = TemporaryFile()

N.savetxt(valid_file, valid, fmt="%.3f")
N.savetxt(test_file, test, fmt="%.3f")

valid_file.seek(0)
test_file.seek(0)


submission = zipfile.ZipFile("pca_"+src+'_'+str(k)+"_add_"+str(add)+"_reduced_train.zip" , "w", compression=zipfile.ZIP_DEFLATED)

submission.writestr("terry_sda_valid.prepro", valid_file.read())
submission.writestr("terry_sda_final.prepro", test_file.read())

submission.close()
valid_file.close()
test_file.close()



