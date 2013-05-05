import sys

_, model_path, niter = sys.argv

niter = int(niter)

from pylearn2.utils import serial

model = serial.load(model_path)

import theano.tensor as T

X = T.matrix()

top = model.hidden_layers[-1]
W = top.W
from pylearn2.utils import sharedX
top.W = sharedX(W.get_value() * 0.)

Q = model.mf(X, niter=niter)

H1, H2, Y = Q
H2, _ = H2

assert H2.ndim == 2

Y_hat = T.dot(H2, W) + top.b

Y = T.matrix()

y_hat = T.argmax(Y_hat, axis=1)
y = T.argmax(Y, axis=1)

misclass = T.neq(y_hat, y).mean()

from theano import function
f = function([X, Y], misclass)

from pylearn2.datasets.mnist import MNIST

dataset = MNIST(which_set='train', one_hot=1, start=50000, stop=60000)

print f(dataset.X, dataset.y)
