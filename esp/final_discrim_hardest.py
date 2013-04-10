from galatea.esp import FinalIm2Word
import numpy as np

dataset = FinalIm2Word(start=00, stop=1000
                          )

from pylearn2.utils import serial

model = serial.load('rectifier_7_best.pkl')

import theano.tensor as T
X = T.matrix()
state = model.fprop(X)
target = T.matrix()
wrong_target = T.matrix()

right_cost = model.layers[-1].kl(Y=target, Y_hat=state)
wrong_cost = model.layers[-1].kl(Y=wrong_target, Y_hat=state)

acc = (wrong_cost > right_cost).mean()

from theano import function

f = function([X, target, wrong_target], acc)

wrong_target = dataset.y.copy()
used = np.zeros((dataset.y.shape[0],), dtype='bool')
minmin = np.inf
for i in xrange(wrong_target.shape[0]):
    dists = np.square(dataset.y - dataset.y[i,:]).sum(axis=1)
    dists[i] = np.inf
    dists[used] = np.inf
    assert dists.min() != 0
    idx = np.argmin(dists)
    used[idx] = 1
    wrong_target[i, :] = dataset.y[idx, :].copy()
    if dists.min() < minmin:
        minmin = dists.min()
        print i, idx, minmin

acc = f(dataset.X, dataset.y, wrong_target)

print acc
