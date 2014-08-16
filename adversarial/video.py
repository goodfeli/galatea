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

Z = np.zeros((endpoints * steps_per_point, nvis))

rng = np.random.RandomState([2014, 6, 11])

start = rng.randn(nvis)
idx = 0
for j in xrange(endpoints):
    stop = rng.randn(nvis)

    for i in xrange(0, steps_per_point):
        alpha = float(i) / float(steps_per_point - 1)
        Z[idx, :] = alpha * stop + (1. - alpha) * start
        idx += 1
    start = stop

from pylearn2.utils import sharedX
Z = sharedX(Z)


if isinstance(space, VectorSpace):
    # For some reason format_as from VectorSpace is not working right
    samples = model.generator.mlp.fprop(Z).eval()

    is_color = samples.shape[-1] % 3 == 0 and samples.shape[-1] != 48 * 48
    samples = dataset.get_topological_view(samples)
else:
    total_dimension = space.get_total_dimension()
    import numpy as np
    num_colors = 1
    if total_dimension % 3 == 0:
        num_colors = 3
    w = int(np.sqrt(total_dimension / num_colors))
    from pylearn2.space import Conv2DSpace
    desired_space = Conv2DSpace(shape=[w, w], num_channels=num_colors, axes=('b',0,1,'c'))
    samples = space.format_as(batch=model.generator.mlp.fprop(Z),
            space=desired_space).eval()
    is_color = samples.shape[-1] == 3

from pylearn2.utils import image
for i in xrange(endpoints * steps_per_point):
    img = samples[i, :, :, :]
    img /= np.abs(img).max()
    img /= 2
    img += 0.5
    number = str(i)
    while len(number) < len(str(endpoints * steps_per_point)):
        number = '0' + number
    path = '/Tmp/video/' + number + '.png'
    image.save(path, img)
