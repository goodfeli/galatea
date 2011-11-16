from pylearn2.utils import serial
import sys
from pylearn2.config import yaml_parse
import numpy as np
from pylearn2.gui import patch_viewer

patch_rescale = False

model_path = sys.argv[1]

model = serial.load(model_path)



from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
theano_rng = RandomStreams(42)
design_examples_var = model.s3c.random_design_matrix(batch_size = model.dbm.negative_chains,
        theano_rng = theano_rng, H_sample = model.dbm.H_chains)
from theano import function
print 'compiling sampling function'
f = function([],design_examples_var)
print 'sampling'
design_examples = f()
print 'loading dataset'
dataset = yaml_parse.load(model.dataset_yaml_src)
examples = dataset.get_topological_view(design_examples)

examples /= np.abs(examples).max()

cols = int(np.sqrt(model.dbm.negative_chains))
rows = model.dbm.negative_chains / cols + 1

assert len(examples.shape) == 4
is_color = examples.shape[3] == 3

pv = patch_viewer.PatchViewer( (rows, cols), examples.shape[1:3], is_color = is_color)

for i in xrange(rows*cols):
        pv.add_patch(examples[i,:,:,:], activation = 0.0, rescale = patch_rescale)

pv.show()
