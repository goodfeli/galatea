from pylearn2.utils import serial
import sys
from pylearn2.config import yaml_parse
from pylearn2.gui.patch_viewer import PatchViewer

_, model_path = sys.argv

print 'Loading model...'
model = serial.load(model_path)

dataset_yaml_src = model.dataset_yaml_src

print 'Loading data (used for setting up visualization and seeding gibbs chain) ...'
dataset = yaml_parse.load(dataset_yaml_src)

rows = 10
cols = 10
m = rows * cols

vis_batch = dataset.get_batch_topo(m)

_, patch_rows, patch_cols, channels = vis_batch.shape

assert _ == m

pv = PatchViewer((rows,cols), (patch_rows,patch_cols), is_color = (channels==3))

def show():
    for i in xrange(m):
        display_batch = dataset.adjust_for_viewer(vis_batch)
        pv.add_patch(display_batch[i,:,:,:], rescale = False)
    pv.show()

print 'showing seed data...'
show()

# Make shared variables representing the sampling state of the model
layer_to_state = model.make_layer_to_state(m)
# Seed the sampling with the data batch
layer_to_state[model.visible_layer].set_value(vis_batch)
