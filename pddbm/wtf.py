import sys
import numpy as np

X = np.load(sys.argv[1])

from theano import shared

X = shared(X[0:10,0:10])

o1 = X.mean()
o2 = X.mean(axis=0).mean()

o = abs(o1-o2)

from theano import tensor as T

G = T.grad(o,X)

from theano import function

f = function([],o,updates = { X : X + .01 * G })

while True:
    print f()

