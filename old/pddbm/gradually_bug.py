C = 1.0

from scipy import io
from pylearn2.models.svm import DenseMulticlassSVM
import numpy as np

train = io.loadmat('pddbm_train_features.mat')
test = io.loadmat('pddbm_test_features.mat')

X, H, G, y = [train[key] for key in ['X','H','G','y']]
Xt, Ht, Gt, yt = [test[key] for key in ['X','H','G','y']]


from pylearn2.datasets.mnist import MNIST
fuck_you = MNIST(which_set = 'train')
assert np.allclose(X, fuck_you.X)
print y.shape
print fuck_you.y.shape
assert np.allclose(y, fuck_you.y)
X = fuck_you.X

X = X[0:1000,:]
y = y[0:1000]

svm_type = DenseMulticlassSVM( C = C)

print X.shape
print 'fitting svm...'
svm = svm_type.fit( X.copy(), y)
print '...done'
print svm.predict(X).shape
assert svm.predict(X).shape[0] == X.shape[0]

def acc(svm, feat, lab):
    print feat.shape
    pred = svm.predict(feat)
    assert pred.shape[0] == feat.shape[0]
    print pred.shape
    print pred.dtype
    print lab.shape
    print lab.dtype
    match = pred == lab
    print match.shape
    print match.dtype
    return match.mean()

print acc(svm, X, y)
print acc(svm, Xt, yt)
