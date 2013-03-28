from pylearn2.utils import serial
import sys

model = serial.load(sys.argv[1])

from matplotlib import pyplot
import numpy as np

norms = np.sqrt(np.square(model.dbm.W[0].get_value()).sum(axis=1))
assert norms.shape == (500,)

pyplot.hist(norms)
pyplot.show()


