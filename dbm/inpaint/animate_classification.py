import numpy as np
from pylearn2.utils import serial
import sys
from theano import function
from pylearn2.config import yaml_parse
from pylearn2.gui.patch_viewer import PatchViewer
from galatea.ui import get_choice
from theano.gof.op import get_debug_values
from theano.printing import min_informative_str

ignore, model_path = sys.argv
m = 10
model = serial.load(model_path)
if hasattr(model,'set_batch_size'):
    model.set_batch_size(m)

dataset = yaml_parse.load(model.dataset_yaml_src)


space = model.get_input_space()
X = space.make_theano_batch()
X.tag.test_value = space.get_origin_batch(m).astype(X.dtype)

inputs = [X]

history = model.mf(X, return_history=True)
for elem in history:
    assert isinstance(elem, (list, tuple))
    assert len(elem) == len(model.hidden_layers)
outputs = [elem[-1] for elem in history]

for elem in outputs:
    for value in get_debug_values(elem):
        if value.shape[0] != m:
            print 'culprint is',id(elem)
            print min_informative_str(elem)
            quit(-1)

f = function(inputs, outputs)


n_classes = model.hidden_layers[-1].n_classes
if isinstance(n_classes, float):
    assert n_classes == int(n_classes)
    n_classes = int(n_classes)
assert isinstance(n_classes, int)
templates = np.zeros((n_classes, space.get_total_dimension()))
for i in xrange(n_classes):
    for j in xrange(-1, -dataset.X.shape[0], -1):
        if dataset.y[j,i]:
            templates[i, :] = dataset.X[j, :]

print 'use test set?'
choice = get_choice({ 'y' : 'yes', 'n' : 'no' })
if choice == 'y':
    dataset = dataset.get_test_set()

topo = X.ndim > 2

while True:
    X, Y = dataset.get_batch_design(m, include_labels = True)
    Xt = dataset.get_topological_view(X)
    if topo:
        X = Xt

    args = [X]

    Y_sequence = f(*args)

    for elem in Y_sequence:
        assert elem.shape[0] == m

    rows = m

    cols = 1+len(Y_sequence)


    pv = PatchViewer((rows, cols), (Xt.shape[1], Xt.shape[2]), is_color = True,
            pad = (8,8) )

    for i in xrange(m):

        #add original patch
        patch = Xt[i,:,:,:].copy()
        patch = dataset.adjust_for_viewer(patch)
        if patch.shape[-1] != 3:
            patch = np.concatenate( (patch,patch,patch), axis=2)
        pv.add_patch(patch, rescale = False, activation = (1,0,0))
        orig_patch = patch

        def label_to_vis(Y_elem):
            prod =  np.dot(Y_elem, templates)
            assert Y_elem.ndim == 1
            rval = np.zeros((1, prod.shape[0]))
            rval[0,:] = prod
            return rval

        # Add the inpainted sequence
        for Y_hat in Y_sequence:
            cur_Y_hat = Y_hat[i,:]
            Y_vis = label_to_vis(cur_Y_hat)
            Y_vis = dataset.adjust_for_viewer(dataset.get_topological_view(Y_vis))
            if Y_vis.ndim == 4:
                assert Y_vis.shape[0] == 1
                Y_vis = Y_vis[0,:,:,:]
            if Y_vis.ndim == 2:
                Y_vis = Y_vis.reshape(Y_vis.shape[0], Y_vis.shape[1], 1)
            if Y_vis.shape[-1] == 1:
                Y_vis = np.concatenate([Y_vis]*3,axis=2)
            pv.add_patch(Y_vis, rescale=False, activation=(0,0,1))


    pv.show()

    print 'Waiting...'
    x = raw_input()
    if x == 'q':
        break
    print 'Running...'
