from pylearn2.datasets.cifar10 import CIFAR10
import time
import warnings
from theano.printing import Print
import numpy as np
from galatea.dbm.inpaint.super_dbm import SuperDBM
from galatea.dbm.inpaint.super_dbm import GaussianConvolutionalVisLayer
from galatea.dbm.inpaint.super_dbm import ConvMaxPool
from galatea.dbm.inpaint.super_dbm import Softmax
from pylearn2.utils import serial
from theano.gof.op import get_debug_values
from theano.printing import min_informative_str

dataset = CIFAR10(which_set = 'train', one_hot = True, gcn = 55.)

rng = np.random.RandomState([2012,07,24])
irange = .05
nvis = 784
nclass = 10
nhid = 500
mf_iter = 10
batch_size = 100
lr = .0005
momentum = 1./20.

from pylearn2.utils import sharedX
import theano.tensor as T

X = dataset.X
y = dataset.y


def make_super_dbm(n_iter):
    return SuperDBM(batch_size = 100,
                    visible_layer = GaussianConvolutionalVisLayer(
                                        rows = 32,
                                        cols = 32,
                                        channels = 3,
                                        init_beta = 1.,
                                        init_mu = 0.),
                    hidden_layers = [
                            ConvMaxPool(output_channels = 64,
                                kernel_rows = 7,
                                kernel_cols = 7,
                                pool_rows = 2,
                                pool_cols = 2,
                                irange = .05,
                                layer_name = 'h0',
                                init_bias = -3.),
                            ConvMaxPool(output_channels = 128,
                                kernel_rows = 4,
                                kernel_cols = 4,
                                pool_rows = 2,
                                pool_cols = 2,
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

#mf1mod = make_super_dbm(1)
mf1mod = serial.load('cifar10_model.pkl')

#for param1, paramn in zip(mf1mod.get_params(),mfnmod.get_params()):
#    param1.set_value(paramn.get_value())
#warnings.warn("parameters are not necessarily identical at start (might be if all the seeds are hardcoded though)")

Xb = mf1mod.visible_layer.get_input_space().make_batch_theano()
Xb.tag.test_value = dataset.get_topological_view(X[5800:5900,:])
Xb.name = 'Xb'
yb = mf1mod.hidden_layers[-1].get_output_space().make_batch_theano()
yb.tag.test_value = y[5800:5900]
yb.name = 'yb'

ymf1 = mf1mod.mf(Xb)[-1]
ymf1.name = 'ymf1'
#ymfn = mfnmod.mf(Xb)[-1]

def log_p_yb(ymf):
    warnings.warn("I handle softmax instability by pulling the argument out of the node, should just go into theano and stabilize log softmax directly")
    assert ymf.ndim == 2
    assert yb.ndim == 2

    arg = ymf.owner.inputs[0]
    arg.name = 'arg'

    arg = arg - arg.max(axis=1).dimshuffle(0,'x')
    arg.name = 'safe_arg'

    unnormalized = T.exp(arg)
    unnormalized.name = 'unnormalized'

    Z = unnormalized.sum(axis=1)
    Z.name = 'Z'

    log_ymf = arg - T.log(Z).dimshuffle(0,'x')

    log_ymf.name = 'log_ymf'

    example_costs =  yb * log_ymf
    example_costs.name = 'example_costs'
    #log P(y = i) = log exp( arg_i ) / sum_j exp( arg_j)
    #           = arg_i - log sum_j exp(arg_j)
    return example_costs.mean()

mf1_cost = - log_p_yb( ymf1 )
mf1_cost.name = 'mf1_cost'
#mfn_cost = - log_p_yb( ymfn )

updates = {}

alpha = T.scalar()
alpha.name = 'alpha'
alpha.tag.test_value = lr

for cost, params in [ (mf1_cost, mf1mod.get_params()) ]:
        #(mfn_cost, mfnmod.get_params()) ]:
    for param in params:
        if param.name != 'mu' and param.name != 'beta':
            inc = sharedX(np.zeros(param.get_value().shape))
            grad = T.grad(cost,param)
            #grad = Print('d cost / d '+param.name,attrs=['min','max'])(grad)
            new_inc = momentum * inc - alpha * grad
            for v in get_debug_values(new_inc):
                assert not np.any(np.isnan(v))
                assert not np.any(np.isinf(v))
            updates[param] = param + new_inc
            #updates[param] = Print('updates['+param.name+']',attrs=['min','max'])(updates[param])
            for v in get_debug_values(updates[param]):
                assert not np.any(np.isnan(v))
                assert not np.any(np.isinf(v))
            updates[inc] = new_inc

from theano import function

func = function([Xb,yb,alpha], updates = updates)

nodes = func.maker.fgraph.toposort()

count = 0
for node in nodes:
    if str(type(node.op)).lower().find('hostfrom') != -1:
        count += 1
    found = 0
    for ipt in node.inputs:
        if ipt.owner is not None and str(type(ipt.owner.op)).lower().find('hostfrom') != -1:
            found += 1
            try:
                print ipt.ndim,'dimensions'
            except:
                print 'no ndm'
            print min_informative_str(ipt)
    if found > 0:
        print type(node.op), found
        try:
            print '\t',type(node.op.scalar_op)
        except:
            pass

print count


"""
i = 58
for key in mf1mod.hidden_layers[0].transformer.get_params():
    func = function([Xb,yb,alpha], updates[key], on_unused_input = 'ignore')

    output = func(dataset.get_topological_view(X[i*batch_size:(i+1)*batch_size,:]),
            y[i*batch_size:(i+1)*batch_size,:],lr)
    if np.any(np.isnan(output)):
        print 'found a nan in the update for '+key.name
        target = updates[key]

        rtopo = [ target ]

        q = 0
        while q < len(rtopo):
            var = rtopo[q]
            if var.owner is not None:
                for ipt in var.owner.inputs:
                    if ipt not in rtopo:
                        rtopo.append(ipt)
            q += 1

        def add_inputs(var,depth = 1):
            print 'depth',depth
            if var.owner is not None:
                for ipt in var.owner.inputs:
                    if ipt not in rtopo:
                        rtopo.append(ipt)
                        add_inputs(ipt,depth+1)
        add_inputs(target)

        for elem in rtopo:
            assert not isinstance(elem,str)

        func = function([Xb,yb,alpha], rtopo)

        outputs = func(dataset.get_topological_view(X[i*batch_size:(i+1)*batch_size,:]),
                y[i*batch_size:(i+1)*batch_size,:],lr)

        last = 0
        for i in xrange(1,len(outputs)):
            if np.any(np.isnan(outputs[i])):
                last = i

        print 'last failing node found at position ',last
        print 'failing node must have been'
        print min_informative_str(rtopo[last])
        print 'type is ',type(rtopo[last])
        print 'values of inputs:'
        for ipt in rtopo[last].owner.inputs:
            for i in xrange(len(rtopo)):
                if ipt is rtopo[i]:
                    print '\t',(outputs[i].min(),outputs[i].max())
                    break
"""


test = CIFAR10(which_set = 'test', one_hot = True, gcn = 55.)


yl = T.argmax(yb,axis=1)

mf1acc = 1.-T.neq(yl , T.argmax(ymf1,axis=1)).mean()
#mfnacc = 1.-T.neq(yl , T.argmax(mfny,axis=1)).mean()

batch_acc = function([Xb,yb],[mf1acc])

def accs():
    mf1_accs = []
    for i in xrange(10000/batch_size):
        mf1_accs.append( batch_acc(test.get_topological_view(test.X[i*batch_size:(i+1)*batch_size,:]),
            test.y[i*batch_size:(i+1)*batch_size,:])[0])
    return sum(mf1_accs) / float(len(mf1_accs))

def taccs():
    mf1_accs = []
    for i in xrange(X.shape[0]/batch_size):
        mf1_accs.append( batch_acc(dataset.get_topological_view(X[i*batch_size:(i+1)*batch_size,:]),
            y[i*batch_size:(i+1)*batch_size,:])[0])
    return sum(mf1_accs) / float(len(mf1_accs))

alpha = lr
print 'running...'
epoch = 1
while True:
    for param in mf1mod.get_params():
        val = param.get_value()
        assert not np.any(np.isnan(val))
        assert not np.any(np.isinf(val))
    serial.save('cifar10_r_model.pkl',mf1mod)
    if epoch % 1 ==0:
        print 'train acc: ',taccs()
    print 'test acc: ',accs()
    print 'doing epoch',epoch

    t1 = time.time()
    for i in xrange(X.shape[0]/batch_size):

        """
        if i ==0:
            i = 58
        else:
            i = 1
        """

        if (i * batch_size) % 10000 == 0:
            print '\t',i*batch_size
            for param in mf1mod.get_params():
                val = param.get_value()
                assert not np.any(np.isnan(val))
                assert not np.any(np.isinf(val))
        #note: calling this with a size-zero slice causes a floating point
        #exception that brings down the interpreter, might want to debug that
        #and add better error handling
        func(dataset.get_topological_view(X[i*batch_size:(i+1)*batch_size,:]),
            y[i*batch_size:(i+1)*batch_size,:],alpha)

    t2 = time.time()

    print 'epoch took',t2-t1

    epoch += 1



