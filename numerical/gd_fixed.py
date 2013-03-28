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

f = 0.5 * T.dot(x,T.dot(A,x)) + T.dot(b,x)

alpha = T.scalar()

g = T.grad(f,x)

update = function([alpha], f, updates = { x : x - alpha * g } )

f = function([],f)

opt_x = np.linalg.solve(A.get_value(),-b.get_value())
x.set_value(opt_x)
print 'Optimal objective:',f()
x.set_value(np.cast[config.floatX](rng.randn(n)))

val, vec = np.linalg.eig(A.get_value())
print 'condition number: ',val.max()/val.min()

t = 1

while True:
    obj =  update( .1 )
    if t % 100000 == 0:
        print t,obj
    t += 1
