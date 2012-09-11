#using nhid = 500
#running this with lr=.01 and no frills results in this getting over 97 % accuracy for both models
#but it takes thousands of epochs
#with lr = .1 by 850 epochs MF-1 has converged to 97.77 and MF-n to 0.9792
#with lr = 1 by 200 epochs both had converged to 97.9
#with lr = 1 and lr_decay = 1.005, MF-1 converged to 0.9795 and MF-N to 0.9806
#using nhid = 1000
from pylearn2.datasets.mnist import MNIST
from theano.printing import Print
import numpy as np
from cRBM import cRBM

dataset = MNIST(which_set = 'train', one_hot = True)

rng = np.random.RandomState([2012,07,24])
irange = .05
nvis = 784
nclass = 10
nhid = 1000
mf_iter = 10
batch_size = 100
lr = 1.
lr_decay = 1.005
init_tv = 1.
tv_mult = 1.
l1wd = .00
l2wd = .001
sp_coeff = .000
sp_targ = .1

W1 = rng.uniform(-irange,irange,(nvis,nhid))
b1 = np.zeros((nhid,))-2.
W2 = rng.uniform(-irange,irange,(nhid,nclass))
b2 = np.zeros((nclass,))

from pylearn2.utils import sharedX
import theano.tensor as T


X = sharedX(dataset.X)
y = sharedX(dataset.y)

idx = T.iscalar()
idx.tag.test_value = 0

Xb = X[idx*batch_size:(idx+1)*batch_size,:]
yb = y[idx*batch_size:(idx+1)*batch_size,:]


mf1mod = cRBM(W1,b1,W2,b2, mf_iter = 1)
mfnmod = cRBM(W1,b1,W2,b2, mf_iter = mf_iter)

ymf1_arg = mf1mod.mf1y_arg(Xb)
ymfn_arg = mfnmod.mfny_arg(Xb)
H1 = mf1mod.mf1H(Xb)
Hn = mf1mod.mfnH(Xb)

def log_p_yb(y_arg):
    assert y_arg.ndim == 2
    mx = y_arg.max(axis=1)
    assert mx.ndim == 1
    y_arg = y_arg - mx.dimshuffle(0,'x')
    assert yb.ndim == 2
    example_costs =  (yb * y_arg).sum(axis=1) - T.log(T.exp(y_arg).sum(axis=1))

    #log P(y = i) = log exp( arg_i ) / sum_j exp( arg_j)
    #           = arg_i - log sum_j exp(arg_j)
    return example_costs.mean()

mf1_cost = - log_p_yb( ymf1_arg) + \
             l1wd * T.sqr(mf1mod.W1).sum() +\
             l2wd * T.sqr(mf1mod.W2).sum() +\
             sp_coeff * abs(H1.mean(axis=0)-sp_targ).sum()
mfn_cost = - log_p_yb( ymfn_arg) +\
             l1wd * T.sqr(mfnmod.W1).sum() +\
             l2wd * T.sqr(mfnmod.W2).sum() +\
             sp_coeff * abs(Hn.mean(axis=0)-sp_targ).sum()

updates = {}

alpha = T.scalar()
alpha.tag.test_value = 1e-4

tv = T.scalar()
momentum = 1. - 1. / tv

for cost, params in [ (mf1_cost, mf1mod.params()),
                      (mfn_cost, mfnmod.params()) ]:
    for param in params:
        inc = sharedX(np.zeros(param.get_value().shape))
        updates[inc] = momentum * inc - alpha * T.grad(cost,param)
        updates[param] = param + updates[inc]

from theano import function

func = function([idx,alpha,tv],[mf1_cost,mfn_cost],updates = updates)

dataset = MNIST(which_set = 'test', one_hot = True)

Xt = sharedX(dataset.X)
yt = sharedX(dataset.y)

mf1yt = mf1mod.mf1y(Xt)
mfnyt = mfnmod.mfny(Xt)

ytl = T.argmax(yt,axis=1)

mf1acc = 1.-T.neq(ytl , T.argmax(mf1yt,axis=1)).mean()
mfnacc = 1.-T.neq(ytl , T.argmax(mfnyt,axis=1)).mean()

accs = function([],[mf1acc,mfnacc])

mf1yb = mf1mod.mf1y(Xb)
mfnyb = mfnmod.mfny(Xb)

ybl = T.argmax(yb,axis=1)

mf1acc = 1.-T.neq(ybl , T.argmax(mf1yb,axis=1)).mean()
mfnacc = 1.-T.neq(ybl , T.argmax(mfnyb,axis=1)).mean()

baccs = function([idx],[mf1acc,mfnacc])


def taccs():
    result = np.zeros((60000/batch_size,2))
    for i in xrange(60000/batch_size):
        result[i,:] = baccs(i)
    return result.mean(axis=0)

from pylearn2.utils import serial

alpha = lr
epoch = 0
tv = init_tv
while True:
    epoch += 1
    if epoch % 10 == 0:
        serial.save('mf1_model.pkl',mf1mod)
        serial.save('mfn_model.pkl',mfnmod)
        print '\tTRAIN acc',taccs()
    print '\tterminal velocity multiplier:',tv
    print '\tlearning rate:',alpha
    print '\ttest acc: ',accs()
    print 'doing epoch',epoch
    for i in xrange(60000/batch_size):
        mf1_cost, mfn_cost = func(i,alpha,tv)
    alpha = max(alpha/lr_decay,5e-5)
    tv = min(tv*tv_mult,10)
        #print mf1_cost, mfn_cost
