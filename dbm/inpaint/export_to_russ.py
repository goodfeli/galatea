import sys
_, path = sys.argv
from pylearn2.utils import serial
model = serial.load(path)
v = model.visible_layer
h1, h2, c = model.hidden_layers
d = {}
d['hidbiases'] = h1.get_biases().reshape(1,500)
d['hidpen'] = h2.get_weights()
d['labbias'] = c.get_biases().reshape(1,10)
d['labpen'] = c.get_weights().T
d['penbiases'] = h2.get_biases().reshape(1,1000)
d['visbiases'] = v.get_biases().reshape(1,784)
d['vishid'] = h1.get_weights()
from scipy import io
io.savemat('/u/goodfeli/galatea/dbm/use_russ_pcd/stitched.mat', d)
