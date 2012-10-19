import numpy as np
from theano import tensor as T
import matplotlib.pyplot as plt
from theano import function

x = T.vector()

y = abs(x-1) + abs(x+1)
z = T.maximum(abs(x)-1,0)*2

dy = T.grad(y.sum(), x)
dz = T.grad(z.sum(), x)

f = function([x], [y, z, dy, dz])


x = np.linspace(-2.,2.,100).astype(x.dtype)
y, z, dy, dz = f(x)

print np.abs(dy - dz).max()

#plt.plot(x,y)
plt.plot(x, dy)
plt.hold(True)
#plt.plot(x,z)
plt.plot(x, dz)
plt.show()
