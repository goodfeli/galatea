import sys

import theano.tensor as T
from theano import function

from pylearn2.datasets.mnist import MNIST
from pylearn2.utils import serial

import numpy as np

_, model_path = sys.argv
model = serial.load(model_path)

model.niter = 11

X = T.matrix()
X.tag.test_value = np.zeros((100,784),dtype='float32')
Y = T.matrix()
Y.tag.test_value = np.zeros((100,10),dtype='float32')

Q = model.mf(V=X, Y=Y)

H2 = Q[-2][-1]

hid, pen, lab = model.hidden_layers

Y_hat = lab.mf_update(state_below = H2)

true = T.argmax(Y, axis=1)
pred = T.argmax(Y_hat, axis=1)
err = T.neq(true, pred)
err_count = err.sum()

errs = function([X,Y], err_count)

total = 0

dataset = MNIST(which_set = 'train', binarize=1, one_hot=True)

for i in xrange(0, 60000, 100):
    x = dataset.X[i:i+100,:].astype(X.dtype)
    assert x.shape == (100, 784)
    y = dataset.y[i:i+100,:].astype(Y.dtype)
    assert y.shape == (100, 10)
    total += errs(x, y)

print total
print total / 60000.
