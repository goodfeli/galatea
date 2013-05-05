print "Loading dataset"
from pylearn2.datasets.cifar10 import CIFAR10
dataset = CIFAR10(which_set='train', axes = ('c', 0, 1, 'b'))

print "Building graph"
rows = 10
cols = 10
m = rows * cols
from pylearn2.space import Conv2DSpace

space = Conv2DSpace([32, 32], num_channels=3, axes=('c', 0, 1, 'b'))

X = space.make_batch_theano()

from galatea.maxout import GCN_C01B2

gcn = GCN_C01B2()
gcn.set_input_space(space)

normed = gcn.fprop(X)

from galatea.maxout import OnlineWhitener

whitener = OnlineWhitener()

whitener.set_input_space(gcn.get_output_space())

white = whitener.fprop(normed)

assert white.ndim == 4

b01c = white.dimshuffle(3, 1, 2, 0)

from theano import function
print "Compiliing"
f = function([X], b01c)
print "Running"
from galatea.maxout import pad
X = pad(dataset=dataset, amt=8).get_batch_topo(m)

yaml_str = """!obj:galatea.maxout.pad {
            amt: 8,
                    dataset: !obj:pylearn2.datasets.zca_dataset.ZCA_Dataset {
                                    preprocessed_dataset: !pkl: "${PYLEARN2_DATA_PATH}/cifar10/pylearn2_gcn_whitened/train.pkl",
                                                preprocessor: !pkl: "${PYLEARN2_DATA_PATH}/cifar10/pylearn2_gcn_whitened/preprocessor.pkl",
                                                            start: 0,
                                                                        stop: 40000,
                                                                                    axes: ['c', 0, 1, 'b']
                                                                                        }}"""
from pylearn2.config.yaml_parse import load

other = load(yaml_str)

other_X = other.get_batch_topo(m)

import numpy as np

new_X = np.zeros((3, 32, 32, m), dtype='float32')
other_new_X = new_X.copy()

rng = np.random.RandomState([2013, 5, 3])
for i in xrange(m):
    r = rng.randint(0,8)
    c = rng.randint(0,8)
    cropped = X[:, r:r+32, c:c+32, i]
    other_cropped = other_X[:, r:r+32, c:c+32, i]
    if rng.randint(2):
        cropped = cropped[:, :, ::-1]
        other_cropped = other_cropped[:, :, ::-1]
    new_X[:,:,:,i] = cropped.copy()
    other_new_X[:,:,:,i] = other_cropped.copy()
X = new_X
other_X = other_new_X


b01c = f(X)

max_abs = max(np.abs(b01c).max(), np.abs(other_X).max())

b01c = b01c / max_abs
other_X = other_X / max_abs

other_X = np.transpose(other_X, (3, 1, 2, 0))

print "Formatting"
from pylearn2.gui.patch_viewer import PatchViewer

pv = PatchViewer(grid_shape=(rows, cols), patch_shape=(32, 32), is_color=True)

for i in xrange(m):
    pv.add_patch(b01c[i,:,:,:], rescale=False)
    pv.add_patch(other_X[i,:,:,:], rescale=False)
print "Showing"
pv.save('/u/goodfeli/vis.png')



