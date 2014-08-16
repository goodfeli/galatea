from pylearn2.utils import serial
import sys
_, model_path = sys.argv
model = serial.load(model_path)
from pylearn2.gui.patch_viewer import make_viewer
space = model.generator.get_output_space()
from pylearn2.space import VectorSpace
from pylearn2.config import yaml_parse
import numpy as np

match_train = True
if match_train:
    dataset = yaml_parse.load(model.dataset_yaml_src)

grid_shape = None

nvis = model.generator.mlp.input_space.get_total_dimension()
endpoints = 50
steps_per_point = 100


rng = np.random.RandomState([2014, 6, 11])

def make_Z():
    start = rng.randn(nvis)
    idx = 0
    rval = np.zeros((endpoints * steps_per_point, nvis))
    for j in xrange(endpoints):
        stop = rng.randn(nvis)

        for i in xrange(0, steps_per_point):
            alpha = float(i) / float(steps_per_point - 1)
            rval[idx, :] = alpha * stop + (1. - alpha) * start
            idx += 1
        start = stop
    return rval

from pylearn2.utils import sharedX
Z = sharedX(make_Z())
assert Z.ndim == 2

from theano import function

if isinstance(space, VectorSpace):
    # For some reason format_as from VectorSpace is not working right
    f = function([], model.generator.mlp.fprop(Z))

    def samples_func():
        samples = f()
        return dataset.get_topological_view(samples)

else:
    total_dimension = space.get_total_dimension()
    import numpy as np
    num_colors = 1
    if total_dimension % 3 == 0:
        num_colors = 3
    w = int(np.sqrt(total_dimension / num_colors))
    from pylearn2.space import Conv2DSpace
    desired_space = Conv2DSpace(shape=[w, w], num_channels=num_colors, axes=('b',0,1,'c'))
    samples_func = function([], space.format_as(batch=model.generator.mlp.fprop(Z),
            space=desired_space))


streams = []

for i in xrange(36):
    print "stream",i
    Z.set_value(make_Z().astype(Z.dtype))
    streams.append(samples_func())

from pylearn2.gui.patch_viewer import PatchViewer
for i in xrange(endpoints * steps_per_point):
    print "file ",i
    viewer = PatchViewer(grid_shape=(6,6), patch_shape=(streams[0].shape[1], streams[0].shape[2]), is_color=False)
    for j in xrange(36):
        viewer.add_patch(streams[j][i, :, :, :] * 2. - 1., rescale=False)
    number = str(i)
    while len(number) < len(str(endpoints * steps_per_point)):
        number = '0' + number
    path = '/Tmp/video/' + number + '.png'
    viewer.save(path)
