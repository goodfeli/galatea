from pylearn2.utils import serial
import sys
from pylearn2.config import yaml_parse
import numpy as np
from pylearn2.gui import patch_viewer

patch_rescale = True

model_path = sys.argv[1]

model = serial.load(model_path)



from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
theano_rng = RandomStreams(42)
assert hasattr(model.dbm,'V_chains') and model.dbm.V_chains is not None

#print model.dbm.V_chains.get_value()

design_examples_var = model.s3c.random_design_matrix(batch_size = model.dbm.negative_chains,
        theano_rng = theano_rng, H_sample = model.dbm.V_chains)
from theano import function
print 'compiling sampling function'
f = function([],design_examples_var)
print 'sampling'
design_examples = f()
print 'loading dataset'
from pylearn2.datasets import control
control.push_load_data(False)
dataset = yaml_parse.load(model.dataset_yaml_src)
examples = dataset.get_topological_view(design_examples)

assert not np.any(np.isnan(examples))
assert not np.any(np.isinf(examples))

mx = np.abs(examples).max()
if mx > 1e-7:
    examples /= mx
assert not np.any(np.isnan(examples))
assert not np.any(np.isinf(examples))

cols = int(np.sqrt(model.dbm.negative_chains))
rows = model.dbm.negative_chains / cols + 1

assert len(examples.shape) == 4
is_color = examples.shape[3] == 3

pv = patch_viewer.PatchViewer( (rows, cols), examples.shape[1:3], is_color = is_color)

for i in xrange(min(model.dbm.negative_chains,rows*cols)):
    patch = examples[i,:,:,:]
    assert not np.any(np.isnan(patch))
    assert not np.any(np.isinf(patch))
    pv.add_patch( dataset.adjust_for_viewer(patch), activation = 0.0, rescale = False)

pv.show()
