import zipfile
from tempfile import TemporaryFile
import numpy as N

bandwidth = 1000.0
k = 50
new_whiten = False
tsne = True
feature_type = 'tri'
#options are 'id' (one-hot code), 'tri' (Adam Coates' "triangle code")

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
test  = N.load(test_file)

#X = N.concatenate((train,valid,test),axis=0)

#normalize X (we forgot to do this in terry_spectral.py)
#X =  (X.T / N.sqrt(N.square(X).sum(axis=1))).T

if new_whiten:
    name += '_whitened'


#valid_start = train.shape[0]
#assert valid_start == 4000
#test_start = valid_start + valid.shape[0]

#valid = X[valid_start:test_start,:]
#test = X[test_start:,:]

mu = N.load(name+'.npy')

assert not new_whiten #todo-- handle loading of whitening params, whitening of data

#del X

def make_features(X):
    m,n = X.shape

    dists = N.zeros((m,k))

    print '\tcomputing distances'
    for i in xrange(k):
        dists[:,i] = N.sqrt(N.square((X - mu[i,:])).sum(axis=1))

    min_dist_inds = dists.argmin(axis=1)

    if feature_type == 'id':
        rval = N.zeros((m,k), dtype=X.dtype)
        for i in xrange(k):
            rval[:,i] = min_dist_inds == i
    elif feature_type == 'tri':
        d = dists.mean(axis=1)
        subdists = (-dists.T +d).T
        rval = subdists * (subdists >= 0.0)
    else:
        assert False       #other feature types not implemented yet

    return rval

print "Extracting features"
valid = make_features(valid)
#test  = make_features(test)
test = valid

print "WARNING: just doing dummy features for test"

print "Saving"
valid_file = TemporaryFile()
test_file = TemporaryFile()

N.savetxt(valid_file, valid, fmt="%.3f")
N.savetxt(test_file, test, fmt="%.3f")

valid_file.seek(0)
test_file.seek(0)

name = name.replace('means','')
submission = zipfile.ZipFile(name+feature_type+".zip" , "w", compression=zipfile.ZIP_DEFLATED)

submission.writestr("terry_sda_valid.prepro", valid_file.read())
submission.writestr("terry_sda_final.prepro", test_file.read())

submission.close()
valid_file.close()
test_file.close()

