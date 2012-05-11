from pylearn2.utils import serial
import sys
from pylearn2.config import yaml_parse
import numpy as np
from pylearn2.gui import patch_viewer
from pylearn2.utils import sharedX

patch_rescale = True

model_path = sys.argv[1]

model = serial.load(model_path)
model.make_pseudoparams()


rows = 10
cols = 10

print 'loading dataset'
dataset = yaml_parse.load(model.dataset_yaml_src)

print 'HACK'
from pylearn2.datasets.mnist import MNIST
dataset = MNIST(which_set = 'test', center = 0, shuffle = True)


init_examples = dataset.get_batch_design( rows * cols )


print 'init_examples ',(init_examples.min(),init_examples.mean(),init_examples.max())

model.dbm.use_cd = False
model.use_cd = False
model.negative_chains = rows * cols
model.dbm.redo_everything() #this just redoes the chains

hidden_obs = model.inference_procedure.infer(sharedX(init_examples))

from theano import function
from theano.tensor.shared_randomstreams import RandomStreams
from pylearn2.models.dbm import Sampler
theano_rng = RandomStreams(17)
sample_from = Sampler(theano_rng)

outputs = [ sample_from(hidden_obs['H_hat']) ]
for G_hat in hidden_obs['G_hat']:
    outputs.append( sample_from(G_hat) )
outputs.append( hidden_obs['S_hat'])
init_chain_hid = function([], outputs)()

model.dbm.V_chains = sharedX(init_chain_hid[0])
model.dbm.H_chains = [ sharedX(init_chain_elem) for init_chain_elem in init_chain_hid[1:-1] ]
S_hat = init_chain_hid[-1]

assert hasattr(model.dbm,'V_chains') and model.dbm.V_chains is not None
design_examples_var = model.s3c.random_design_matrix(batch_size = rows * cols,
        theano_rng = theano_rng, H_sample = model.dbm.V_chains)
print 'compiling sampling function'
f = function([],design_examples_var)
g = function([],model.s3c.random_design_matrix(batch_size = rows * cols,
            theano_rng = theano_rng, H_sample = model.dbm.V_chains, S_sample = S_hat))
sample = function([], updates = model.dbm.get_sampling_updates())


examples = dataset.get_topological_view(init_examples)
assert len(examples.shape) == 4
is_color = examples.shape[3] == 3
pv = patch_viewer.PatchViewer( (rows, cols), examples.shape[1:3], is_color = is_color)

first = True

while True:

    assert not np.any(np.isnan(examples))
    assert not np.any(np.isinf(examples))

    if first:
        print 'Showing seed examples...'

    examples = dataset.adjust_for_viewer(examples)

    for i in xrange(rows*cols):
        patch = examples[i,:,:,:]
        assert not np.any(np.isnan(patch))
        assert not np.any(np.isinf(patch))
        pv.add_patch(patch, activation = 0.0, rescale = False)#patch_rescale)

    pv.show()

    print 'waiting...'
    x = raw_input()
    if x == 'q':
        quit()

    if first:
        first = False
        second = True
        print 'Showing sample from posterior, using S_hat from inference for s'

        design_examples = g()
    else:
        if second:
            second = False
            print 'Showing sample from posterior'
        else:
            try:
                reps = int(x)
            except:
                reps = 1
            print reps,'step(s) of sampling...'

            for i in xrange(reps):
                sample()
        design_examples = f()
    examples = dataset.get_topological_view(design_examples)

