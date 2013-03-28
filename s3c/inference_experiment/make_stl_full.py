from pylearn2.utils import serial

data = serial.load('${STL10_WHITENED_TRAIN}')

X = data.get_batch_design(1)

X = data.get_batch_design(500)

import numpy as np

np.save('data/stl/full.npy',X)


