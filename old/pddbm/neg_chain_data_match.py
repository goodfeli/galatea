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
        theano_rng = theano_rng, H_sample = model.dbm.V_chains, full_sample = False)
from theano import function
print 'compiling sampling function'
f = function([],design_examples_var)
print 'sampling'
design_examples = f()
print 'loading dataset'
from pylearn2.datasets import control
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

pv = patch_viewer.PatchViewer( (rows, 2*cols), examples.shape[1:3], is_color = is_color)

print 'normalizing data...'
X = dataset.X
X = X.T
X /= np.sqrt(np.square(X).sum(axis=0)+.01)
X = X.T

for i in xrange(min(model.dbm.negative_chains,rows*cols)):
    print 'matching example',i
    patch = examples[i,:,:,:]
    assert not np.any(np.isnan(patch))
    assert not np.any(np.isinf(patch))

    blah = design_examples[i,:]
    blah /= np.sqrt(np.square(blah).sum()+.01)

    match = np.dot(X,blah)

    idx = np.argmax(match)

    pv.add_patch( patch, activation = 1.0, rescale = True)

    patch = dataset.get_topological_view(X[idx:idx+1,:])[0,:,:,:]

    pv.add_patch( patch, activation = 0.0, rescale = True)

pv.show()


