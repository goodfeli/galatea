from theano import tensor as T
from theano import function
from pylearn2.utils import sharedX
import numpy as np
from theano import config

rng = np.random.RandomState([1,2,3])

n = 3

src = rng.randn(n,n)
A = sharedX(np.dot(src.T,src))
b = sharedX(rng.randn(n))
x = sharedX(rng.randn(n))
inc = sharedX(np.zeros((n,)))

f = 0.5 * T.dot(x,T.dot(A,x)) + T.dot(b,x)

alpha = T.scalar()
momentum = T.scalar()

g = T.grad(f,x)

cur_inc = momentum * inc + (1.-momentum) *  ( - alpha * g)

update = function([alpha, momentum], f, updates = { x : x + cur_inc, inc : cur_inc } )

f = function([],f)

opt_x = np.linalg.solve(A.get_value(),-b.get_value())
x.set_value(opt_x)
print 'Optimal objective:',f()
x.set_value(np.cast[config.floatX](rng.randn(n)))

val, vec = np.linalg.eig(A.get_value())
print 'condition number: ',val.max()/val.min()

t = 1

while True:
    obj =  update( max(1e-5, 1./float(t)), .5 )
    if t % 100000 == 0:
        print t,obj
    t += 1
