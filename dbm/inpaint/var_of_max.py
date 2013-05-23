import numpy as np

rng = np.random.RandomState([1,2,3])
x = []
y = []

for i in xrange(1, 100):
    x.append(i)

    samples = rng.randn(i,100000)

    maxes = samples.max(axis=0)
    y.append(maxes.std(axis=0))

from matplotlib import pyplot
pyplot.plot(x,y)
pyplot.show()
