import numpy as np
from pylearn2.datasets.cifar100 import CIFAR100

print 'loading cifar100...'
y = CIFAR100(which_set = 'train').y_fine

print 'loading full'
X = np.load('/data/lisatmp2/goodfeli/bet_the_farm.npy')

print 'loading restricted'
Z = np.load('/data/lisatmp2/goodfeli/hack.npy')

idx = 0
for i in xrange(50000):
    if y[i] < 2:
        cur_X = X[i,:,:,:]
        cur_Z = Z[idx,:,:,:]
        diffs = cur_X - cur_Z
        max_diff = np.abs(diffs).max()
        print i,'\t',max_diff
        idx += 1
