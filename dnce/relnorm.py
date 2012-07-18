dim =  []
y = []
import numpy as np
rng = np.random.RandomState([1,2,3])
m = 100

for d in xrange(1,1000):
    X = rng.randn(m,d)
    Y = rng.randn(m,d) / 10.
    norm_X = np.sqrt(np.square(X).sum(axis=1))
    norm_Y = np.sqrt(np.square(Y).sum(axis=1))
    ratio = norm_Y / norm_X
    mn = ratio.mean()

    dim.append(d)
    y.append(mn)

from matplotlib import pyplot
pyplot.plot(dim,y)
pyplot.show()
