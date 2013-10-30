#show_samples.py hacked to be convenient for working remotely
from pylearn2.utils import serial
import sys
from pylearn2.config import yaml_parse
import numpy as np
from pylearn2.gui import patch_viewer
from pylearn2.utils import sharedX

patch_rescale = True

ignore, model_path, reps, out = sys.argv

reps = int(reps)


model = serial.load(model_path)
model.make_pseudoparams()


rows = 10
cols = 10

print 'loading dataset'
dataset = yaml_parse.load(model.dataset_yaml_src)

init_examples = dataset.get_batch_design( rows * cols )

print 'init_examples ',(init_examples.min(),init_examples.mean(),init_examples.max())

model.dbm.use_cd = False
model.use_cd = False
model.negative_chains = rows * cols
model.dbm.redo_everything() #this just redoes the chains

hidden_obs = model.inference_procedure.infer(sharedX(init_examples))

from theano import function
outputs = [ hidden_obs['H_hat'] ]
for G_hat in hidden_obs['G_hat']:
    outputs.append(G_hat)
init_chain_hid = function([], outputs)()

model.dbm.V_chains = sharedX(init_chain_hid[0])
model.dbm.H_chains = [ sharedX(init_chain_elem) for init_chain_elem in init_chain_hid[1:] ]

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
theano_rng = RandomStreams(42)
assert hasattr(model.dbm,'V_chains') and model.dbm.V_chains is not None
design_examples_var = model.s3c.random_design_matrix(batch_size = rows * cols,
        theano_rng = theano_rng, H_sample = model.dbm.V_chains)
print 'compiling sampling function'
f = function([],design_examples_var, updates = model.dbm.get_sampling_updates())


print 'init_examples later',(init_examples.min(),init_examples.mean(),init_examples.max())
examples = dataset.get_topological_view(init_examples)
print 'examples ',(examples.min(),examples.mean(),examples.max())
assert len(examples.shape) == 4
is_color = examples.shape[3] == 3
pv = patch_viewer.PatchViewer( (rows, cols), examples.shape[1:3], is_color = is_color)

first = True
while True:

    assert not np.any(np.isnan(examples))
    assert not np.any(np.isinf(examples))

    examples = dataset.adjust_for_viewer(examples)

    if first:
        first = False
    else:
        for i in xrange(rows*cols):
            patch = examples[i,:,:,:]
            assert not np.any(np.isnan(patch))
            assert not np.any(np.isinf(patch))
            pv.add_patch(patch, activation = 0.0, rescale = False)#patch_rescale)

        pv.save(out)

        break

    for i in xrange(reps):
        design_examples = f()
    examples = dataset.get_topological_view(design_examples)
