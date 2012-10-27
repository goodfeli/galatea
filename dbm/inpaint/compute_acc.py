from galatea.dbm.inpaint.super_dbm import SuperDBM
from galatea.dbm.inpaint.super_dbm import GaussianConvolutionalVisLayer
from galatea.dbm.inpaint.super_dbm import ConvMaxPool
from galatea.dbm.inpaint.super_dbm import Softmax
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
import sys

_, model_path = sys.argv

model = serial.load(model_path)

src = model.dataset_yaml_src
batch_size = model.force_batch_size
# handle bug in older pkl files, where set_batch_size had updated
# batch_size but not force_batch_size
if hasattr(model, 'batch_size') and model.batch_size != model.force_batch_size:
    batch_size = model.batch_size


assert src.find('train') != -1
test = yaml_parse.load(src)
x = raw_input("test acc?")
if x == 'y':
    test = test.get_test_set()
    assert test.X.shape[0] == 10000
else:
    assert x == 'n'

if x == 'y':
    if not (test.X.shape[0] == 10000):
        print test.X.shape[0]
        assert False
else:
    # compute the train accuracy on what the model
    # was trained on, not the entire train set
    assert test.X.shape[0] in [40000,50000]

test.X = test.X.astype('float32')
test.y = test.y.astype('float32')

import theano.tensor as T

def make_super_dbm(n_iter):
    return SuperDBM(batch_size = 100,
                    visible_layer = GaussianConvolutionalVisLayer(
                                        rows = 32,
                                        cols = 32,
                                        channels = 3,
                                        init_beta = 1.,
                                        init_mu = 0.),
                    hidden_layers = [
                            ConvMaxPool(output_channels = 256,
                                kernel_rows = 6,
                                kernel_cols = 6,
                                pool_rows = 3,
                                pool_cols = 3,
                                irange = .05,
                                layer_name = 'h0',
                                init_bias = -3.),
                            ConvMaxPool(output_channels = 128,
                                kernel_rows = 4,
                                kernel_cols = 4,
                                pool_rows = 3,
                                pool_cols = 3,
                                irange = .01,
                                layer_name = 'h1',
                                init_bias = -3.),
                            #ConvMaxPool(output_channels = 3200,
                            #    kernel_rows = 5,
                            #    kernel_cols = 5,
                            #    pool_rows = 1,  #really inefficient way of getting a non-pooled sigmoid layer
                            #    pool_cols = 1,
                            #    irange = .01,
                            #    layer_name = 'h2',
                            #    init_bias = -1.),
                            #ConvMaxPool(output_channels = 1600,
                            #    kernel_rows = 3, #really inefficient way of getting a densely connected sigmoid layer
                            #    kernel_cols = 3,
                            #    pool_rows = 1,
                            #    pool_cols = 1,
                            #    irange = .01,
                            #    layer_name = 'h3',
                            #    init_bias = -1.),
                            Softmax(n_classes = 10,
                                irange = .01)
                        ],
                    niter = n_iter)

Xb = model.get_input_space().make_batch_theano()
Xb.name = 'Xb'
yb = model.hidden_layers[-1].get_output_space().make_batch_theano()
yb.name = 'yb'

ymf = model.mf(Xb)[-1]
ymf.name = 'ymf'

from theano import function

yl = T.argmax(yb,axis=1)

mf1acc = 1.-T.neq(yl , T.argmax(ymf,axis=1)).mean()

batch_acc = function([Xb,yb],[mf1acc])


def accs():
    mf1_accs = []
    for i in xrange(test.X.shape[0]/batch_size):
        print i
        x_arg = test.X[i*batch_size:(i+1)*batch_size,:]
        if Xb.ndim > 2:
            x_arg = test.get_topological_view(x_arg)
        mf1_accs.append( batch_acc(x_arg,
            test.y[i*batch_size:(i+1)*batch_size,:])[0])
    return sum(mf1_accs) / float(len(mf1_accs))


result = accs()


print result
