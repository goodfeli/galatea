from galatea.esp import FinalIm2Word
import numpy as np

dataset = FinalIm2Word(start=0, stop=500
                          )

from pylearn2.utils import serial

model = serial.load('logreg.pkl')

import theano.tensor as T
X = T.matrix()
state = model.fprop(X)
target = T.matrix()
wrong_target = T.matrix()

right_cost = model.layers[-1].kl(Y=target, Y_hat=state)
wrong_cost = model.layers[-1].kl(Y=wrong_target, Y_hat=state)

from theano.printing import Print
right_cost = Print('right_cost')(right_cost)

acc = (wrong_cost > right_cost).mean()

from theano import function

f = function([X, target, wrong_target], acc)

wrong_target = dataset.y.copy()
used = np.zeros((500,), dtype='bool')
for i in xrange(wrong_target.shape[0]):
    dists = np.square(dataset.y - dataset.y[i,:]).sum(axis=1)
    dists[i] = np.inf
    dists[used] = np.inf
    idx = np.argmin(dists)
    used[idx] = 1
    wrong_target[i, :] = dataset.y[idx, :].copy()

acc = f(dataset.X, dataset.y, wrong_target)
print dataset.y.sum()
print wrong_target.sum()

print acc
