import theano.tensor as T
from theano import function, config
import numpy as N
floatX = config.floatX

z = T.matrix()
x = z + 1.
y = T.mean(x)

f = function([z],T.grad(y,x))

print f(N.cast['float32'](N.random.randn(2,2)))

