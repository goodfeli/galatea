from pylearn2.models.svm import DenseMulticlassSVM
from pylearn2.datasets.cifar100 import CIFAR100
import numpy as np

print 'Loading dataset...'
dataset = CIFAR100( which_set = 'train')
X = np.load('/data/lisatmp2/goodfeli/C3_lab_3_mean.npy')
#print 'h only'
#X = X[:,:,:,0:1600]
#print 'g only'
#X = X[:,:,:,1600:]
X = X.reshape(X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])

print 'Restricting dataset...'
y = dataset.y_fine


Xtr = X[0:1000,:]
ytr = y[0:1000]

Xv = X[1000:2000,:]
yv = y[1000:2000]

Xte = X[2000:3000,:]
yte = y[2000:3000]

bva = 0.

for C in [ 0.001, 0.01, 0.1, 1., 10., 100., 1000., 1e4 ]:
    print 'Fitting svm'
    svm = DenseMulticlassSVM( kernel = 'linear', C = C).fit(Xtr,ytr)

    print 'Valid accuracy'
    acc =  (svm.predict(Xv) == yv).mean()
    print acc

    if acc > bva:
        bva = acc
        print 'Test accuracy'
        ate =  (svm.predict(Xte) == yte).mean()
        print ate
print 'Test accuracy'
print ate


