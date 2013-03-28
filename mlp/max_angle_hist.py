from pylearn2.utils import serial
import sys
from matplotlib import pyplot
import numpy as np

ignore, model_path, layer_idx = sys.argv
layer_idx = int(layer_idx)

model = serial.load(model_path)

layer = model.layers[layer_idx]
k = layer.pool_size
assert layer.pool_stride == k

W, = layer.transformer.get_params()
W = W.get_value()

dots = []

for i in xrange(layer.detector_layer_dim // k):

    ws = [W[:, k*i + j] for j in xrange(k)]
    cur_dots = []
    for a in xrange(k):
        for b in xrange(a+1,k):
            w1 = ws[a]
            w2 = ws[b]
            d1 = w1 / np.sqrt(np.square(w1).sum())
            d2 = w2 / np.sqrt(np.square(w2).sum())
            cur_dots.append(np.dot(d1, d2))
    dots.append(max(cur_dots))

pyplot.hist(dots)
pyplot.show()
