from pylearn2.utils import serial
from pylearn2.utils import sharedX
from pylearn2.config import yaml_parse
from theano import function
import sys
import numpy as np
from pylearn2.gui.patch_viewer import PatchViewer
from galatea.dbm.inpaint import super_dbm


ignore, model_path = sys.argv

model = serial.load(model_path)

# Get access to the intermediate layers of the augmented DBM
if hasattr(model, 'super_dbm'):
    model = model.super_dbm


if hasattr(model,'dataset_yaml_src'):
    dataset = yaml_parse.load(model.dataset_yaml_src)
else:
    from pylearn2.datasets.cifar10 import CIFAR10
    dataset = CIFAR10(which_set = 'test', gcn = 55.)


batch_size = 25
model.set_batch_size(batch_size)
perc = .99
num_examples = 50000
num_layers = len(model.hidden_layers)
num_filters = []
act_record = []
for i in xrange(num_layers):
    layer = model.hidden_layers[i]
    if isinstance(layer, super_dbm.ConvMaxPool):
        num_filters.append(model.hidden_layers[i].output_channels)
    else:
        num_filters.append(layer.detector_layer_dim / layer.pool_size)
    n = num_filters[-1]
    layer_act_record = np.zeros((num_examples,n),dtype='float32')
    act_record.append(layer_act_record)

#make act_func. should return num_layers tensors of shape batch_size, num_filters
print 'making act_func...'
X = model.get_input_space().make_theano_batch()
topo = X.ndim != 2
H_hat = model.mf(X)
acts = []
for layer, state in zip(model.hidden_layers, H_hat):
    p, h = state

    if isinstance(layer, super_dbm.ConvMaxPool):
        p_shape = layer.get_output_space().shape
        i = p_shape[0] / 2
        j = p_shape[1] / 2

        acts.append( p[:,:,i,j] )
    else:
        acts.append(p)
act_func = function([X], acts)
print '...done'


for ii in xrange(0, num_examples, batch_size):
    print 'example',ii
    print '\tcomputing acts'
    X = dataset.X[ii:ii+batch_size, :]
    if topo:
        X = dataset.get_topological_view(X)
    acts = act_func(X)

    print '\trecording acts'
    for layer_idx in xrange(num_layers):
        act_record[layer_idx][ii:ii+batch_size] = acts[layer_idx]


for layer_idx in xrange(num_layers):
    print 'Making viewer for layer',layer_idx
    X = dataset.get_batch_topo(1)
    _, rows, cols, channels = X.shape
    assert _ == 1

    n = num_filters[layer_idx]

    c = int(np.sqrt(n))
    if c % 2 != 0:
        c += 1
    r =  n / c
    if r * c <  n:
        r += 1

    pv = PatchViewer((r,c), (rows, cols), is_color = channels == 3)

    for filter_idx in xrange(num_filters[layer_idx]):
        print '\t\tComputing response image for filter',filter_idx
        filter_act_record = act_record[layer_idx][:, filter_idx]
        filter_act_record = [ (val, idx) for idx, val in enumerate(filter_act_record) ]
        filter_act_record.sort()
        num_top_filters = 1
        thresh = perc * filter_act_record[-1][0]
        while num_top_filters < len(filter_act_record) and filter_act_record[-num_top_filters][0] > thresh:
            num_top_filters += 1
        num_top_filters -= 1
        print '\t\t\tUsing %d top filters' % num_top_filters
        top = filter_act_record[-num_top_filters:]
        idxs = [ elem[1] for elem in top]
        coeffs = [ elem[0] for elem in top]
        coeffs = np.asarray(coeffs)
        assert len(dataset.X.shape) == 2
        batch = dataset.X[idxs,:]
        assert len(batch.shape) == 2
        batch = (coeffs * batch.T).T
        mean = batch.mean(axis=0)
        standard_error = batch.std(axis=0) / np.sqrt(num_top_filters)
        final = mean - standard_error
        final *= mean > standard_error
        img = batch[0:,:].copy()
        img[0,:] = final
        img = dataset.get_topological_view(img)
        img = dataset.adjust_for_viewer(img)
        img = img[0,:,:,:]
        pv.add_patch(img, rescale = False)

    pv.show()


