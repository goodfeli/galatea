from pylearn2.utils import serial

data = serial.load('${STL10_PATCHES_6x6}')

X = data.get_batch_design(1)

X = data.get_batch_design(100000)

import numpy as np

np.save('data/stl/patch.npy',X)


