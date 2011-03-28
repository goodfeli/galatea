import numpy as N
import scipy.linalg as L

bandwidth = 1000.0
k = 50
tsne = True
new_whiten = False

tsne_str = ""
if tsne:
    tsne_str = "_tsne"
prefix = 'terry'+tsne_str+'_spectral_bandwidth_'+str(bandwidth)+'k_'+str(k)
name = prefix + '_means'
train_file = prefix + 'train.npy'
valid_file = prefix + 'valid.npy'
test_file = prefix + 'test.npy'

#train = N.load(train_file)
valid = N.load(valid_file)
#test  = N.load(test_file)

print 'WARNING: only making means based on valid'

X = valid
#X = N.concatenate((train,valid,test),axis=0)

#normalize X (we forgot to do this in terry_spectral.py)
X =  (X.T / N.sqrt(N.square(X).sum(axis=1))).T

m,n = X.shape

print str(m)+' examples with '+str(n)+' features'

if new_whiten:
    print 'whitening'

    print '\tsubtracting mean'
    whitening_mean = X.mean(axis=0)
    X -= whitening_mean


    print '\tfiltering'
    std = X.std(axis=0)
    std[ std < 0.0 ] = 0.0
    filter = N.nonzero(std == 0.0)
    X = X[:, std != 0.0 ]
    n = X.shape[1]

    name += '_whitened'


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

    serial.save(name+'_whitening_params.pkl',(whitening_mean,W,filter))


    mean = X.mean(axis=0)
    print 'mean range: '+str((mean.min(),mean.max()))
    cov = N.dot(X.T,X) /float(m-1)
    print 'cov diagonal range: '+str((cov.diagonal().min(),cov.diagonal().max()))
    for i in xrange(n):
        cov[i,i] = 0
    print 'cov off-diagonal max: '+str(cov.max())



rng = N.random.RandomState([1,2,3])

print 'computing initial clusters'
mu = N.zeros((k,n))
for i in xrange(k):
    mu[i,:] = X[i:m:k,:].mean(axis=0)

dists = N.zeros((m,k))

k_orig = k

killed_on_prev_iter = False
old_kills = {}

iter = 0
while True:
    print "iter "+str(iter)

    if N.sum(N.isnan(mu)) > 0:
        print 'nan'
        quit()

    print '\tcomputing distances'
    for i in xrange(k):
        dists[:,i] = N.square((X - mu[i,:])).sum(axis=1)

    if iter > 0:
        prev_mmd = mmd

    min_dists = dists.min(axis=1)

    mmd = min_dists.mean()

    print 'mean minimum distance: '+str(mmd)

    if iter > 0 and (not killed_on_prev_iter) and mmd == prev_mmd:
        print 'converged'
        break

    print '\tfinding minimum distances'
    min_dist_inds = dists.argmin(axis=1)

    print '\tcomputing means'
    i = 0
    blacklist = []
    new_kills = {}
    killed_on_prev_iter = False
    while i < k:
        b = min_dist_inds == i
        if not N.any(b):
            """  old code let a cluster die if it became empty
            new_mu = N.zeros((k-1,n))
            new_mu[:i,:] = mu[:i,:]
            new_mu[i:,:] = mu[i+1:,:]
            mu = new_mu
            k -= 1
            min_dist_inds -= 1
            print '\tkilled cluster '+str(i)+', '+str(k)+' clusters remain'
            """

            killed_on_prev_iter = True

            #new code initializes empty cluster to be the mean of the d data points farthest from their corresponding means
            if i in old_kills:
                d = old_kills[i]-1
                if d == 0:
                    d = 50
                new_kills[i] = d
            else:
                d = 5
            mu[i,:] = 0
            for j in xrange(d):
                idx = N.argmax(min_dists)
                min_dists[idx] = 0
                print '\tchose point '+str(idx)
                mu[i,:] += X[idx,:]
                blacklist.append(idx)
            mu[i,:] /= float(d)
            print 'cluster '+str(i)+' was empty, reset it to '+str(d)+' far out data points'
            print '\t...recomputing distances for this cluster'
            dists[:,i] = N.square((X-mu[i,:])).sum(axis=1)
            min_dists = dists.min(axis=1)
            for idx in blacklist:
                min_dists[idx] = 0
            min_dist_inds = dists.argmin(axis=1)
            print '\tdone'
            i += 1
        else:
            mu[i,:] = N.mean( X[b,: ] ,axis=0)
            if N.sum(N.isnan(mu)) > 0:
                print min_dist_inds == i
                print 'nan at '+str(i)
                quit()
            i += 1

    old_kills = new_kills

    N.save(name,mu)

    iter += 1
