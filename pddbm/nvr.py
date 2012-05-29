import sys
from pylearn2.utils import serial
import theano.tensor as T
from pylearn2.config import yaml_parse
from theano import function
from pylearn2.gui.patch_viewer import PatchViewer
import numpy as np
assert hasattr(np,'sqrt')

ignore, model_path = sys.argv


print 'loading model'
model = serial.load(model_path)
model.make_pseudoparams()

norms = None
if hasattr(model,'dbm'):
    W2 = model.dbm.rbms[0].transformer._W.get_value()
    norms = np.sqrt( np.square(W2).sum(axis=1))

V = T.matrix()

H_hat = model.infer(V)['H_hat']

mx = H_hat.max(axis=0)
assert mx.ndim == 1
mn = H_hat.min(axis=0)
assert mn.ndim == 1
ranges = mx - mn

print 'loading dataset'
dataset = yaml_parse.load(model.dataset_yaml_src)

X = dataset.get_batch_design(100)

print 'compiling and running function'
ranges = function([V],ranges)(X)
assert len(ranges.shape) == 1

from matplotlib import pyplot as plt

plt.scatter(ranges, norms)
plt.xlabel('range')
plt.ylabel('norm')

plt.show()
