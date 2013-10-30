C = 10.
add_h = False
use_h = False
add_g = False
use_g = False
use_y = False
add_y = False
use_dbm = True

from scipy import io
from pylearn2.models.svm import DenseMulticlassSVM
import numpy as np
import time

train = io.loadmat('pddbm_train_features.mat')
test = io.loadmat('pddbm_test_features.mat')

X, H, G, y = [train[key] for key in ['X','H','G','y']]
Xt, Ht, Gt, yt = [test[key] for key in ['X','H','G','y']]
assert yt.shape[1] == 1
if use_dbm:
    dbm_feat = io.loadmat('dbm_feat.mat')
    if not np.allclose(X,dbm_feat['train_X']):
        #print X.min()
        #print X.mean()
        #print X.max()
        #print dbm_feat['train_X'].min()
        #print dbm_feat['train_X'].mean()
        #print dbm_feat['train_X'].max()
        print np.abs(X-dbm_feat['train_X']).max()
        assert False
    assert np.allclose(Xt,dbm_feat['test_X']).max()
    X = dbm_feat['train_X']
    H = dbm_feat['train_H']
    G = dbm_feat['train_G']
    Xt = dbm_feat['test_X']
    Ht = dbm_feat['test_H']
    Gt = dbm_feat['test_G']

from pylearn2.datasets.mnist import MNIST
dataset = MNIST( which_set = 'train' )
assert np.allclose(dataset.X,X)
assert np.allclose(dataset.y,y[:,0])
X = dataset.X
y = dataset.y
dataset = MNIST( which_set = 'test' )
assert np.allclose(dataset.X,Xt)
assert np.allclose(dataset.y,yt[:,0])
Xt = dataset.X
yt = dataset.y

if add_h:
    assert not use_dbm
    X = np.concatenate( (X,H), axis=1)
    Xt = np.concatenate( (Xt,Ht), axis=1)

if use_h:
    X = H
    Xt = Ht

if use_g:
    X = G
    Xt = Gt

if add_g:
    X = np.concatenate( (X,G), axis=1)
    Xt = np.concatenate( (Xt,Gt), axis=1)

if use_y:
    X = train['Y']
    Xt = test['Y']

if add_y:
    X = np.concatenate((X,train['Y']),axis=1)
    Xt = np.concatenate((Xt,test['Y']),axis=1)

if len(y.shape) != 1:
    assert len(y.shape) == 2
    assert y.shape[1] == 1
    y = y[:,0]
if len(yt.shape) != 1:
    assert len(yt.shape) == 2
    assert yt.shape[1] == 1
    yt = yt[:,0]




Xsub = X[0:50000,:]
ysub = y[0:50000]

Xv = X[50000:,:]
yv = y[50000:]



assert Xt.shape[1] == X.shape[1]
assert Xsub.shape[1] == X.shape[1]
assert Xv.shape[1] == X.shape[1]

print 'fitting svm...'
t1 = time.time()
svm = DenseMulticlassSVM( C = C, kernel = 'linear').fit( Xsub, ysub)
t2 = time.time()
print '...done, took',t2-t1,'seconds'

def acc(svm, feat, lab):
    t1 = time.time()
    pred = svm.predict(feat)
    t2 = time.time()
    print 'svm prediction took',t2-t1,'seconds'
    assert pred.shape[0] == feat.shape[0]
    match = pred == lab
    assert pred.shape == lab.shape
    assert pred.dtype == lab.dtype
    for i in xrange(int(lab.min()),int(lab.max())+1):
        if (pred == i).sum() == 0:
            print "warning: Never predicted class "+str(i)
    for i in xrange(int(pred.min()),int(pred.max())+1):
        assert (lab == i).sum() > 0
    return match.mean()

acc_sub = acc(svm, Xsub, ysub)
print 'subtrain acc',acc_sub


valid_acc = acc(svm,Xv,yv)
print 'valid acc ', valid_acc

#print 'train acc',acc(svm, X, y)
print 'test acc (trained only on sub)', acc(svm, Xt, yt)
