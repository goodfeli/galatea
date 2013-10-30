import sys
from pylearn2.utils import serial

model = serial.load(sys.argv[1])

W = model.W.get_value()

import numpy as np

norms = np.sqrt(1e-8+np.square(W).sum(axis=0))

from matplotlib import pyplot as plt

plt.hist(norms)
plt.show()
