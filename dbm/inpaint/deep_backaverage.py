from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from theano import function
import sys
import numpy as np
from pylearn2.gui.patch_viewer import PatchViewer
from galatea.dbm.inpaint import super_dbm


if len(sys.argv) == 2:
    ignore, model_path = sys.argv
    data_override = None
else:
    ignore, model_path, data_override = sys.argv



model = serial.load(model_path)

# Get access to the intermediate layers of the augmented DBM
if hasattr(model, 'super_dbm'):
    model = model.super_dbm


if hasattr(model,'dataset_yaml_src'):
    dataset = yaml_parse.load(model.dataset_yaml_src)
else:
    from pylearn2.datasets.cifar10 import CIFAR10
    dataset = CIFAR10(which_set = 'test', gcn = 55.)

rng = np.random.RandomState([2012,10,24])
if data_override == 'binary_noise':
    dataset.X = rng.uniform(0., 1., dataset.X.shape) > 0.5
elif data_override == 'gaussian_noise':
    dataset.X = rng.randn( * dataset.X.shape).astype(dataset.X.dtype)


batch_size = 25
model.set_batch_size(batch_size)
perc = .99
num_examples = 50000
num_layers = len(model.hidden_layers)
num_filters = []
act_record = []
def add_filters_and_act_record(layer):
    if 'CompositeLayer' in str(type(layer)):
        for sublayer in layer.components:
            add_filters_and_act_record(sublayer)
    else:
        if isinstance(layer, super_dbm.ConvMaxPool):
            num_filters.append(layer.output_channels)
        elif isinstance(layer, super_dbm.Softmax):
            num_filters.append(layer.n_classes)
        else:
            num_filters.append(layer.detector_layer_dim / layer.pool_size)
        n = num_filters[-1]
        layer_act_record = - np.ones((num_examples,n),dtype='float32')
        act_record.append(layer_act_record)

for i in xrange(num_layers):
    layer = model.hidden_layers[i]
    add_filters_and_act_record(layer)
num_layers = len(num_filters)

#make act_func. should return num_layers tensors of shape batch_size, num_filters
print 'making act_func...'
X = model.get_input_space().make_theano_batch()
topo = X.ndim != 2
H_hat = model.mf(X)
acts = []

def add_acts(layer, state):

    if isinstance(layer, super_dbm.CompositeLayer):
        for sublayer, substate in zip(layer.components, state):
            add_acts(sublayer, substate)
    elif isinstance(layer, super_dbm.ConvMaxPool):
        p, h = state
        p_shape = layer.get_output_space().shape
        i = p_shape[0] / 2
        j = p_shape[1] / 2

        acts.append( p[:,:,i,j] )
    elif isinstance(layer, super_dbm.Softmax):
        acts.append(state)
    else:
        p, h = state
        acts.append(p)


for layer, state in zip(model.hidden_layers, H_hat):
    add_acts(layer, state)

act_func = function([X], acts)
print '...done'


data_mean = dataset.X.mean(axis=0)

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
        assert filter_act_record.min() >= 0.0
        filter_act_record = [ (val, idx) for idx, val in enumerate(filter_act_record) ]
        filter_act_record.sort()
        num_top_filters = 1
        peak = filter_act_record[-1][0]
        thresh = perc * peak
        while num_top_filters < len(filter_act_record) and filter_act_record[-num_top_filters][0] > thresh:
            num_top_filters += 1
        top = filter_act_record[-num_top_filters:]
        print '\t\t\tUsing %d top examples (activation value %f - %f)' % \
                (num_top_filters, top[0][0], peak)
        idxs = [ elem[1] for elem in top]
        coeffs = [ elem[0] for elem in top]
        coeffs = np.asarray(coeffs)

        if coeffs.sum() == 0.0:
            print 'WARNING: skipping totally inactive unit.'
            continue

        assert len(dataset.X.shape) == 2
        batch = dataset.X[idxs,:]
        assert len(batch.shape) == 2
        mean = batch.mean(axis=0)


        def weighted_mean(batch, coeffs):
            return (batch.T * coeffs).mean(axis=1)

        def weighted_std(batch, coeffs, mean):
            deviations = batch - mean
            sq_deviations = np.square(batch-mean)
            weighted_variance = (sq_deviations.T * coeffs).mean(axis=1)
            return np.sqrt(weighted_variance)

        mean = weighted_mean(batch, coeffs)
        std = weighted_std(batch, coeffs, mean)
        standard_error = std / np.sqrt(coeffs.sum())

        assert not np.any(np.isnan(standard_error))
        #print 'mean     ',mean[0:5]
        #print 'std      ',std[0:5]
        #print 'stderr   ',standard_error[0:5]

        def clip_below(x, mn):
            mask = (x > mn)
            return x * mask + (1-mask)*mn

        def clip_above(x, mx):
            mask = x < mx
            return mask * x + (1-mask) * mx

        final = (mean > data_mean) * clip_below(mean - standard_error, data_mean) + \
                (mean <= data_mean) * clip_above(mean + standard_error, data_mean)

        #print 'data mean',data_mean[0:5]
        #print 'final    ',final[0:5]

        assert final.ndim == 1
        img = final.reshape(1,final.shape[0])
        img = dataset.get_topological_view(img)
        img = dataset.adjust_for_viewer(img)
        img = img[0,:,:,:]
        pv.add_patch(img, rescale=False, activation=peak)

    pv.show()


