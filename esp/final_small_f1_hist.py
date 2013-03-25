from galatea.esp import FinalSmallIm2Word

valid = FinalSmallIm2Word(
                          )

from pylearn2.utils import serial

model = serial.load('rectifier_7.pkl')

import theano.tensor as T
X = T.matrix()
state = model.fprop(X)
target = T.matrix()


y_hat = state > 0.5
y = target > 0.5

y = T.cast(y, state.dtype)
y_hat = T.cast(y_hat, state.dtype)

tp = (y * y_hat).sum(axis=0)
fp = ((1-y) * y_hat).sum(axis=0)
precision = tp / T.maximum(1., tp + fp)

recall = tp / T.maximum(1., y.sum(axis=0))

f1 = 2. * precision * recall / T.maximum(1, precision + recall)

from theano import function

ttp = tp.sum()
tpredp = y.sum()

tpr = ttp / T.maximum(1., tpredp)
trr = ttp / T.maximum(1., y.sum())

total_f1 = 2. * tpr * trr / T.maximum(1., tpr + trr)


f = function([X, target], [f1, total_f1])

f1, total_f1 = f(valid.X, valid.y)

print 'overall f1: ',total_f1
print 'mean per class f1: ',f1.mean()

from matplotlib import pyplot

pyplot.hist(f1, bins=100)

print 'showing...'
pyplot.show()
print 'done showing'

