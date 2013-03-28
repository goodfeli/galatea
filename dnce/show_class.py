from pylearn2.utils import serial
import sys
from pylearn2.gui.patch_viewer import PatchViewer
from pylearn2.config import yaml_parse
from pylearn2.utils import sharedX
from theano import function
import theano.tensor as T
import numpy as np

ignore, model_path = sys.argv

model = serial.load(model_path)

dataset = yaml_parse.load(model.dataset_yaml_src)

rows = 10
cols = 5

X = dataset.get_batch_design(rows * cols )

noise = model.dnce.noise_conditional

assert noise.is_symmetric()

Y = function([],noise.random_design_matrix(sharedX(X)))()

prob = function([],T.nnet.sigmoid(model.free_energy(sharedX(Y))-model.free_energy(sharedX(X))))()
assert prob.ndim == 1

Xt = dataset.get_topological_view(X)
Yt = dataset.get_topological_view(Y)
Xt = dataset.adjust_for_viewer(Xt)
Yt = dataset.adjust_for_viewer(Yt)

pv = PatchViewer( (rows, cols * 2), Xt.shape[1:3], is_color = Xt.shape[-1] == 3)

for i in xrange(Xt.shape[0]):
    assert prob[i] >= 0.0
    assert prob[i] <= 1.0
    assert not np.isnan(prob[i])
    assert not np.isinf(prob[i])
    pv.add_patch(Xt[i,:,:,:], activation = prob[i])
    pv.add_patch(Yt[i,:,:,:], activation = 1. - prob[i])

pv.show()
