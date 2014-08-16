from pylearn2.utils import serial
import sys
_, model_path = sys.argv
model = serial.load(model_path)
g = model.generator
g = g.mlp
n = g.get_input_space().get_total_dimension()
import numpy as np
rng = np.random.RandomState([1, 2, 3])
u1 = rng.randn(n)
u1 /= np.sqrt(np.square(u1).sum())
u2 = rng.randn(n)
u2 = u2 - u1 * np.dot(u2, u1)
u2 /= np.sqrt(np.square(u2).sum())
rows = 10
cols = 10
Z = np.zeros((rows * cols, n))
idx = 0
for i in xrange(rows):
    alpha = float(i) / float(rows-1)
    x = (1-alpha) * -1. + alpha * 1.
    for j in xrange(cols):
        alpha = float(j) / float(cols - 1)
        y = (1-alpha) * -1. + alpha * 1.
        Z[idx, :] = x * u1 + y * u2
        idx += 1
from pylearn2.utils import sharedX
Z = sharedX(Z)
X = g.fprop(Z).eval()
if X.shape[-1] == 100:
    X = np.transpose(X, (3, 1, 2, 0))
from pylearn2.gui.patch_viewer import make_viewer
v = make_viewer(X)
v.show()
