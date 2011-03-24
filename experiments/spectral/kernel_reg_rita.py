import zipfile
from tempfile import TemporaryFile
import numpy as N
import serialutil
from auc import embed
import scipy.linalg as L

k = 256


#valid = serialutil.load('/data/lisatmp/ift6266h11/rita-pca-lcn/newdata_pca_lcn/valid1000.pkl')
#test =  serialutil.load('/data/lisatmp/ift6266h11/rita-pca-lcn/newdata_pca_lcn/test1000.pkl' )
valid = N.loadtxt('/u/goodfeli/rita/rita_sda_valid.prepro')
test  = N.loadtxt('/u/goodfeli/rita/rita_sda_final.prepro')



valid = N.concatenate((valid,test),axis=0)

whiten = True
if whiten:
    X = valid

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

    valid = X
#



print 'warning, test is just dummy features'

valid_feat = N.zeros((valid.shape[0],valid.shape[0]))
test = valid_feat

for i in xrange(valid.shape[0]):
    print i
    diffs = valid - valid[i,:]
    dists = N.sqrt(N.square(diffs).sum(axis=1))

    vals = list(dists)
    vals = sorted(vals)
    thresh = vals[k]

    valid_feat[:,i] = dists

valid_feat = valid_feat.mean() - valid_feat
valid_feat = valid_feat * ( valid_feat >= 0.0 )


print valid_feat.sum(axis=0)


valid = valid_feat[0:4096,:]
#test = valid_feat[4096:,:]

#from auc import embed

print embed.score(valid_feat, N.concatenate((N.zeros((4096,1)),N.ones((4096,1)))) )


"""

print "Saving"
valid_file = TemporaryFile()
test_file = TemporaryFile()

N.savetxt(valid_file, valid, fmt="%.3f")
N.savetxt(test_file, test, fmt="%.3f")

valid_file.seek(0)
test_file.seek(0)

submission = zipfile.ZipFile("rita_tsne_triangle_"+str(k)+"nn.zip" , "w", compression=zipfile.ZIP_DEFLATED)

submission.writestr("terry_sda_valid.prepro", valid_file.read())
submission.writestr("terry_sda_final.prepro", test_file.read())

submission.close()
valid_file.close()
test_file.close()
"""
