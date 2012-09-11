#batch size 30, nhid = 500, 500 init_lr = .1 lr_decay = 1.005 resulted in this getting to about 88 / 93 by 100 epochs
#batch size 10, nhid = 500, 500 init_lr = .1 lr_decay = 1.01, init_tv = 2, tv_mult = 1.01 resulted in both getting stuck in the 80s
#batch size 10, nhid = 500, 500 init_lr = .01 lr_decay = 1.01, init_tv = 2, tv_mult = 1.01 resulted in both getting stuck in the 80s
#same but init_lr = 1., lr_decay 1.05, init_tv =1, tv_mult = 1.01 resulted in train acc of 0.87935     0.98371667 after a few hundred epochs, at which point lr had saturated to minimum value
#same but lr_decay = 1.04 resulted in
#epoch 1649
#   TRAIN acc [ 0.87958333  0.99083333]
#   terminal velocity multiplier: 10.0
#   learning rate: 5e-05
#   test acc:  [array(0.8878), array(0.9687)]
#
# bumped up layer 2 init bias from -2 to -1
#this resulted in
#doing epoch 459
#        TRAIN acc [ 0.87928333  0.98866667]
#                terminal velocity multiplier: 10.0
#                        learning rate: 5e-05
#                                test acc:  [array(0.8881), array(0.97)]
# decreased lr_decay to 1.01
# blew up after epoch 231 first time I ran it. re-ran
#doing epoch 2829
#        TRAIN acc [ 0.87956667  0.99425   ]
#                terminal velocity multiplier: 10.0
#                        learning rate: 5e-05
#                                test acc:  [array(0.8883), array(0.9729)]
# decreased lr_decay to 1.001 and increased mn_lr from 5e-5 to 5e-4 because
#it seems like optimization was getting stuck above.
# also, noticed that there are lots of
# duplicate filters in first layer. bumped init bias from -2 to -1
# and dropped init_lr to .1 trying to combat this
# resulted in
#doing epoch 3959
#        TRAIN acc [ 0.87911667  0.97428333]
#        terminal velocity multiplier: 10.0
#        learning rate: 0.00191199723273
#        test acc:  [array(0.8861), array(0.9627)]
#something went bad. I think the low initial learning rate is important
#to not get duplicate filters, but reverting all else
# lr_decay = 1.01
# (left min_lr alone because it was never reached)
# first layer init bias = -2
# This resulted in training acc oscillating rather than converging so I
# also backed min_lr out, back to
#5e-5
# this resulted in
# doing epoch 889
#        TRAIN acc [ 0.87923333  0.99336667]
#                terminal velocity multiplier: 10.0
#                        learning rate: 5e-05
#
# raised layer 1 init_bias to -1 resulted in
#doing epoch 3489
#        TRAIN acc [ 0.87885     0.99271667]
#                terminal velocity multiplier: 10.0
#                        learning rate: 5e-05
#                                test acc:  [array(0.886), array(0.9694)]
#tried turning momentum off
#
#resulted in
#  doing epoch 12959
#        TRAIN acc [ 0.87871667  0.99641667]
#                terminal velocity multiplier: 1.0
#                        learning rate: 5e-05
#                                test acc:  [array(0.8868), array(0.973)]
#running it with
#all weight decay = 0
# lr = .1
# lr_decay = 1.0001
# init_tv = 2.
# tv_mult = 1.001
# max_tv = 10.
# init biases for layer 1 and 2 = -1
#resulted in
#doing epoch 1609
#        TRAIN acc [ 0.99998333  0.09915   ]
#                terminal velocity multiplier: 9.98758894307
#                learning rate: 0.0851384053471
#                test acc:  [array(0.9841), array(0.10089999999999999)]
#tried bringing all weight decay back to 1e-3 to see if that's
#sufficient to salvage the recurrent net / destroy the feedforward net
#looks like it was
#ran it with all weight decay set to 1e-6
#doing epoch 3799
#        TRAIN acc [ 0.99966667  0.9988    ]
#                terminal velocity multiplier: 10.0
#                        learning rate: 0.0022435701647
#                                test acc:  [array(0.9837), array(0.979)]
#
# recurrent still doesn't fit the training set as tightly as
# the feedforward can. that the train acc going up by .0024
# resulted in the test acc going by .005 so it's still probably
# worth trying to fit tighter
#
# set all weight decay to 1e-10
#resulted in
#doing epoch 30619
#        TRAIN acc [ 0.99988333  0.99851667]
#                terminal velocity multiplier: 10.0
#                        learning rate: 5e-05
#                                test acc:  [array(0.9838), array(0.9786)]
# so feedback does seem to result in some optimization problem,
# we can't get the recurrent net to fit the training set as well as the
# simple feed-forward
# the recurrent net also seems to generalize worse for the same level
# of training accuracy


from pylearn2.datasets.mnist import MNIST
from theano.printing import Print
import numpy as np
from cDBM import cDBM

dataset = MNIST(which_set = 'train', one_hot = True)

rng = np.random.RandomState([2012,07,24])
irange = .05
nvis = 784
nclass = 10
nhid1 = 500
nhid2 = 500
mf_iter = 10
batch_size = 100
lr = .1
lr_decay = 1.001
min_lr = 5e-5
init_tv = 2.
tv_mult = 1.001
max_tv = 10.
l1wd = 1e-10
l2wd = 1e-10
l3wd = 1e-10
sp_coeff = .000
sp_targ = .1

W1 = rng.uniform(-irange,irange,(nvis,nhid1))
b1 = np.zeros((nhid1,))-1.
W2 = rng.uniform(-irange,irange,(nhid1,nhid2))
b2 = np.zeros((nhid2,))-1.
W3 = rng.uniform(-irange,irange,(nhid2,nclass))
b3 = np.zeros((nclass,))

from pylearn2.utils import sharedX
import theano.tensor as T


X = sharedX(dataset.X)
y = sharedX(dataset.y)

idx = T.iscalar()
idx.tag.test_value = 0

Xb = X[idx*batch_size:(idx+1)*batch_size,:]
yb = y[idx*batch_size:(idx+1)*batch_size,:]


mf1mod = cDBM(W1,b1,W2,b2, W3, b3,  mf_iter = 1)
mfnmod = cDBM(W1,b1,W2,b2, W3, b3, mf_iter = mf_iter)

ymf1_arg = mf1mod.mf1y_arg(Xb)
ymfn_arg = mfnmod.mfny_arg(Xb)

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
             l3wd * T.sqr(mf1mod.W3).sum()

mfn_cost = - log_p_yb( ymfn_arg) +\
             l1wd * T.sqr(mfnmod.W1).sum() +\
             l2wd * T.sqr(mfnmod.W2).sum() +\
             l3wd * T.sqr(mfnmod.W3).sum()


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
    alpha = max(alpha/lr_decay,min_lr)
    tv = min(tv*tv_mult,max_tv)
        #print mf1_cost, mfn_cost
