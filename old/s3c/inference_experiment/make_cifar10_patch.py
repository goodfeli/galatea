from pylearn2.utils import serial

data = serial.load('${CIFAR10_PATCHES_6x6}')

X = data.get_batch_design(1)

X = data.get_batch_design(100000)

import numpy as np

np.save('data/cifar10/patch.npy',X)


