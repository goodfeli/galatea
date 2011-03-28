import zipfile
import copy
from tempfile import TemporaryFile
import gzip
from pylearn.datasets import utlc
import numpy as N
import scipy
import scipy.sparse
import scipy.linalg as L
import theano


init_bandwidth = 0.6
#bandwidth_step = 0.1
min_hits = 8
whiten = True


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
train = train[0:4000,:]
assert train.shape[0] == 4000
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

train = train[0:4000,:]

print 'concatenating datasets'
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

    W = N.sqrt(m-1) * N.dot(vecs , N.dot( D , vecs.T))

    print '\tprocessing data'
    X = N.dot(X,W)
#



print "HACK, PROJECTING TO UNIT SPHERE AFTER WHITENING"
X = (X.T / N.sqrt(N.square(X).sum(axis=1) )).T

endsets = []

def get_assmts(D):
    global endsets

    m,n = D.shape

    rval = [None for x in range(m) ]

    for i in xrange(m):
        print '\texample '+str(i+1)+'/'+str(D.shape[0])

        iter = 0
        first = True

        pos = D[i,:]

        while True:
            iter += 1
            #print '\t\titer '+str(iter)
            dists = N.sqrt(N.square(X - pos).sum(axis=1))

            bandwidth = init_bandwidth

            hits = N.nonzero( dists <= bandwidth )[0]
            nh = hits.shape[0]

            if nh < min_hits:
                tempdists = dists.copy()
                for i in xrange(min_hits):
                    bandwidth = tempdists.min()
                    tempdists[bandwidth == tempdists] = tempdists.max()
                #
            #

            hits = N.nonzero(dists <= bandwidth)[0]
            nh = hits.shape[0]

            assert nh >= min_hits       

            if first:
                first = False
            else:
                if hits.shape[0] == prev_hits.shape[0] and N.all(hits == prev_hits):
                    break
                #
            #

            pos = X[hits,:].mean(axis=0)
            #print pos

            prev_hits = hits
        print '\t\tfinished after '+str(iter)+' iters, with '+str(hits.shape[0])+' hits'
        # close while loop

        shares_point_with = []
        for endset in endsets:
            for k in xrange(hits.shape[0]):
                if N.any(hits[k] == endset) :
                    assert type(endset) != type((1,2))
                    shares_point_with.append(endset)
                    break
                #
            #close for k
        #close for endset

        if len(shares_point_with) == 0:
            rval[i] = hits
            assert type(hits) != type((1,2))
            endsets.append(hits)
            print '\t\tcreated new cluster, now there are '+str(len(endsets))
        elif len(shares_point_with) == 1:
            #join a cluster
            rval[i] = shares_point_with[0]
            print '\t\tjoined cluster '+str(id(shares_point_with[0]))

            found = False
            for k in xrange(hits.shape[0]):
                if not N.any(hits[k] == shares_point_with[0]):
                    found = True
                    break
            if found:
                points = []
                for k in xrange(shares_point_with[0].shape[0]):
                    points.append(shares_point_with[0][k])
                for k in xrange(hits.shape[0]):
                    if hits[k] not in points:
                        points.append(hits[k])
                points = sorted(points)
                
                old_id = id(shares_point_with[0])
                endsets = [ endset for endset in endsets if id(endset) is not old_id ]
                newhits = N.asarray(points)
                for j in xrange(i):
                    if id(rval[i]) == old_id:
                        rval[i] = newhits
                    #
                #
                endsets.append(newhits)
        else:
            #cause clusters to merge
            
            ids = [id(x) for x in shares_point_with ]
            endsets = [ endset for endset in endsets if id(endset) not in ids ]
            

            newlist = []
            for endset in shares_point_with:
                for k in xrange(endset.shape[0]):
                    if endset[k] not in newlist:
                        newlist.append(endset[k])
                    #
                #
            #
            newlist = sorted(newlist)
            
            newhits = N.asarray(newlist)
            for j in xrange(i):
                if id(rval[i]) in ids:
                    rval[i] = newhits
            rval[i] = newhits
            endsets.append(newhits)
            print "\t\tmerged clusters, now there are "+str(len(endsets))
        # closes if

    # close for i
    return rval
# close get_assmts

valid_assmts = get_assmts(valid)
test_assmts = get_assmts(test)

num_assmts = len(endsets)

def get_features(assmts):
    assmt_ids = N.zeros( (len(assmts),) , dtype='uint32')
    for i in xrange(num_assmts):
        for j in xrange(len(assmts)):
            if endsets[i] is assmts[j]:
                assmt_ids[j] = i
            #
        #
    #

    rval = N.zeros((len(assmts),num_assmts))
    for i in xrange(num_assmts):
        rval[:,i] = assmt_ids == i
    return rval

valid = get_features(valid_assmts)
test_assmts = get_features(test_assmts)


print 'saving'


#Save zip format for direct upload of embedding to server
valid_file = TemporaryFile()
test_file = TemporaryFile()

N.savetxt(valid_file, valid, fmt="%.3f")
N.savetxt(test_file, test, fmt="%.3f")

valid_file.seek(0)
test_file.seek(0)

whiten_str = ""
if whiten:
    whiten_str = "_whitened"

submission = zipfile.ZipFile("terry_mean_shift_bandwidth"+str(init_bandwidth)+whiten_str+".zip" , "w", compression=zipfile.ZIP_DEFLATED)

submission.writestr("terry_sda_valid.prepro", valid_file.read())
submission.writestr("terry_sda_final.prepro", test_file.read())

submission.close()
valid_file.close()
test_file.close()



