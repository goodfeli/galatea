from galatea.esp import Im2Word

dataset = Im2Word(start=99000, stop=100000
                          )

from pylearn2.utils import serial

model = serial.load('rectifier_7.pkl')

import theano.tensor as T
X = T.vector()
state = model.fprop(X.dimshuffle('x', 0))
target = T.matrix()

kl = model.layers[-1].kl(Y=target, Y_hat=state)

from theano import function

f = function([X, target], kl)

ranks = []
for i in xrange(1000):
    kls = f(dataset.X[i,:], dataset.y)
    rank = (kls < kls[i]).sum()
    ranks.append(rank)
    print rank
print sum(ranks) / 1000.

