from pylearn2.utils import serial
import sys
from pylearn2.config import yaml_parse
from pylearn2.gui.patch_viewer import PatchViewer
import time
from theano import function
from theano.sandbox.rng_mrg import MRG_RandomStreams
import numpy as np
import warnings

rows = 10
cols = 10
m = rows * cols

_, model_path = sys.argv

print 'Loading model...'
model = serial.load(model_path)
model.set_batch_size(m)


dataset_yaml_src = model.dataset_yaml_src

print 'Loading data (used for setting up visualization and seeding gibbs chain) ...'
dataset = yaml_parse.load(dataset_yaml_src)


vis_batch = dataset.get_batch_topo(m)

_, patch_rows, patch_cols, channels = vis_batch.shape

assert _ == m

mapback = hasattr(dataset, 'mapback_for_viewer')

pv = PatchViewer((rows,cols*(1+mapback)), (patch_rows,patch_cols), is_color = (channels==3))

def show():
    """
    for ch in xrange(3):
        v_ch = vis_batch[:,:,:,ch]
        pos_ch = vis_batch[:,:,:,ch] * (vis_batch[:,:,:,ch] > 0.)
        abs_ch = np.abs(vis_batch[:,:,:,ch])
        print ch, (v_ch.min(), v_ch.mean(), v_ch.max())
        print ch, 'abs', (abs_ch.min(), abs_ch.mean(), abs_ch.max())
        print ch, 'pos', (pos_ch.min(), pos_ch.mean(), pos_ch.max())
    """
    display_batch = dataset.adjust_for_viewer(vis_batch)
    if display_batch.ndim == 2:
        display_batch = dataset.get_topological_view(display_batch)
    if mapback:
        design_vis_batch = vis_batch
        if design_vis_batch.ndim != 2:
            design_vis_batch = dataset.get_design_matrix(design_vis_batch)
        mapped_batch_design = dataset.mapback_for_viewer(design_vis_batch)
        mapped_batch = dataset.get_topological_view(mapped_batch_design)
    for i in xrange(rows):
        row_start = cols * i
        for j in xrange(cols):
            pv.add_patch(display_batch[row_start+j,:,:,:], rescale = False)
            if mapback:
                pv.add_patch(mapped_batch[row_start+j,:,:,:], rescale = False)
    pv.show()


if hasattr(model.visible_layer, 'beta'):
    beta = model.visible_layer.beta.get_value()
#model.visible_layer.beta.set_value(beta * 100.)
    print 'beta: ',(beta.min(), beta.mean(), beta.max())

print 'showing seed data...'
show()


# Make shared variables representing the sampling state of the model
layer_to_state = model.make_layer_to_state(m)
# Seed the sampling with the data batch
vis_sample = layer_to_state[model.visible_layer]

if vis_sample.ndim == 4:
    vis_sample.set_value(vis_batch)
else:
    vis_sample.set_value(dataset.get_design_matrix(vis_batch))

theano_rng = MRG_RandomStreams(2012+9+18)

# Do one round of clamped sampling so the seed data gets to have an influence
# The sampling is bottom-to-top so if we don't do an initial round where we
# explicitly clamp vis_sample, its initial value gets discarded with no influence
sampling_updates = model.get_sampling_updates(layer_to_state, theano_rng,
        layer_to_clamp = { model.visible_layer : True } )

t1 = time.time()
sample_func = function([], updates=sampling_updates)
t2 = time.time()
print 'Clamped sampling function compilation took',t2-t1
sample_func()


# Now compile the full sampling update
sampling_updates = model.get_sampling_updates(layer_to_state, theano_rng)
assert layer_to_state[model.visible_layer] in sampling_updates

t1 = time.time()
sample_func = function([], updates=sampling_updates)
t2 = time.time()

print 'Sampling function compilation took',t2-t1

while True:
    print 'Displaying samples. How many steps to take next? (q to quit, ENTER=1)'
    while True:
        x = raw_input()
        if x == 'q':
            quit()
        if x == '':
            x = 1
            break
        else:
            try:
                x = int(x)
                break
            except:
                print 'Invalid input, try again'

    for i in xrange(x):
        print i
        sample_func()

    vis_batch = vis_sample.get_value()
    show()


