from pylearn2.utils import serial
import sys
from matplotlib import pyplot
import numpy as np

ignore, model_path, layer_idx = sys.argv
layer_idx = int(layer_idx)

model = serial.load(model_path)

layer = model.layers[layer_idx]
assert layer.pool_size == 2
assert layer.pool_stride == 2

W, = layer.transformer.get_params()
W = W.get_value()

dots = []

for i in xrange(layer.detector_layer_dim // 2):

    w1 = W[:, 2*i]
    w2 = W[:, 2*i+1]
    d1 = w1 / np.sqrt(np.square(w1).sum())
    d2 = w2 / np.sqrt(np.square(w2).sum())
    dots.append(np.dot(d1, d2))

pyplot.hist(dots)
pyplot.show()
