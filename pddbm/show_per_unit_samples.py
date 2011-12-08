from pylearn2.utils import serial
import sys
from pylearn2.config import yaml_parse
import numpy as np
from pylearn2.gui import patch_viewer
from pylearn2.utils import as_floatX
from pylearn2.datasets import control

patch_rescale = True

model_path = sys.argv[1]

model = serial.load(model_path)

nh = model.dbm.rbms[0].nhid
reps = 10

G_sample = np.zeros((nh*reps,nh))

for i in xrange(nh):
    G_sample[i*reps:(i+1)*reps,i] = 1

G_sample = as_floatX(G_sample)

H_prob = model.dbm.inference_procedure.infer_H_hat_one_sided(other_H_hat = G_sample,\
        W = model.dbm.W[0], b = model.dbm.bias_vis)

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
theano_rng = RandomStreams(42)
H_sample = theano_rng.binomial(size = H_prob.shape, n = 1, p = H_prob, dtype = H_prob.dtype)
design_examples_var = model.s3c.random_design_matrix(batch_size = nh * reps,
        theano_rng = theano_rng, H_sample = H_sample)
from theano import function
print 'compiling sampling function'
f = function([],design_examples_var)
print 'sampling'
design_examples = f()
print 'loading dataset'
control.push_load_data(False)
dataset = yaml_parse.load(model.dataset_yaml_src)
examples = dataset.get_topological_view(design_examples)

examples /= np.abs(examples).max()

cols = reps
rows = nh

assert len(examples.shape) == 4
is_color = examples.shape[3] == 3

pv = patch_viewer.PatchViewer( (rows, cols), examples.shape[1:3], is_color = is_color)

for i in xrange(min(examples.shape[0],rows*cols)):
        pv.add_patch(examples[i,:,:,:], activation = 0.0, rescale = patch_rescale)

pv.show()

pv.save('out.png')
