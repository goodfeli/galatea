import theano.tensor as T
from theano import pp

a = T.scalar()
a.name = 'a'

b = a
b.name = 'b'

f = a + T.sqr(b)

g = T.grad(f,a,consider_constant = [b])


print pp(g)





