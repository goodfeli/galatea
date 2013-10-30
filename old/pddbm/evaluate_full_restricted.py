from pylearn2.models.svm import DenseMulticlassSVM
from pylearn2.datasets.cifar100 import CIFAR100
import numpy as np

print 'Loading dataset...'
dataset = CIFAR100( which_set = 'train')
X = np.load('/data/lisatmp2/goodfeli/bet_the_farm.npy')
X = X.reshape(50000, 3 * 3 * 4800)

print 'Restricting dataset...'
y = dataset.y_fine
X = X[y < 2,:]
y = y[y < 2 ]

assert X.shape[0] == 1000
assert y.shape == (1000,)

Xtr = X[0:500,:]
ytr = y[0:500]

Xte = X[500:,:]
yte = y[500:]

print 'Fitting svm'
svm = DenseMulticlassSVM( kernel = 'linear', C = 1.0).fit(Xtr,ytr)

print 'Training accuracy'
print (svm.predict(Xtr) == ytr).mean()

print 'Test accuracy'
print (svm.predict(Xte) == yte).mean()
