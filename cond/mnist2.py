#both models got to upper 97% accuracy before I killed this
#the feed-forward one was still learning but very slowly
#the recurrent one had been oscillating for a long time but
#got to the upper 97% accuracy regime much earlier
#basically, the recurrent version behaved like it had a higher learning
#rate
from pylearn2.datasets.mnist import MNIST
from theano.printing import Print
import numpy as np

dataset = MNIST(which_set = 'train', one_hot = True)

rng = np.random.RandomState([2012,07,24])
irange = .05
nvis = 784
nclass = 10
nhid = 500
mf_iter = 10
batch_size = 100
lr = 1e-2

W1 = rng.uniform(-irange,irange,(nvis,nhid))
b1 = np.zeros((nhid,))-2.
W2 = rng.uniform(-irange,irange,(nhid,nclass))
b2 = np.zeros((nclass,))

from pylearn2.utils import sharedX
import theano.tensor as T

class cRBM:
    def __init__(self,W1, b1,W2,b2):
        self.W1 = sharedX(W1)
        self.W2 = sharedX(W2)
        self.b1 = sharedX(b1)
        self.b2 = sharedX(b2)

    def params(self):
        return [ self.W1, self.W2, self.b1, self.b2 ]

    def mf1y(self, X):
        H = T.nnet.sigmoid(T.dot(X,self.W1)+self.b1)
        y = T.nnet.softmax(T.dot(H,self.W2)+self.b2)
        #y = Print('y')(y)
        return y

    def mf1y_arg(self, X):
        H = T.nnet.sigmoid(T.dot(X,self.W1)+self.b1)
        return T.dot(H,self.W2)+self.b2

    def mfny(self,X):
        H = T.nnet.sigmoid(T.dot(X,2*self.W1)+self.b1)
        y = T.nnet.softmax(T.dot(H,self.W2)+self.b2)

        for i in xrange(mf_iter-1):
            H = T.nnet.sigmoid(T.dot(X,self.W1)+T.dot(y,self.W2.T)+self.b1)
            y = T.nnet.softmax(T.dot(H,self.W2)+self.b2)
        return y

    def mfny_arg(self,X):
        H = T.nnet.sigmoid(T.dot(X,2*self.W1)+self.b1)
        y = T.nnet.softmax(T.dot(H,self.W2)+self.b2)

        for i in xrange(mf_iter-1):
            H = T.nnet.sigmoid(T.dot(X,self.W1)+T.dot(y,self.W2.T)+self.b1)
            y = T.nnet.softmax(T.dot(H,self.W2)+self.b2)
        return T.dot(H,self.W2) + self.b2

X = sharedX(dataset.X)
y = sharedX(dataset.y)

idx = T.iscalar()
idx.tag.test_value = 0

Xb = X[idx*batch_size:(idx+1)*batch_size,:]
yb = y[idx*batch_size:(idx+1)*batch_size,:]


mf1mod = cRBM(W1,b1,W2,b2)
mfnmod = cRBM(W1,b1,W2,b2)

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

C = .001

mf1_cost = - log_p_yb( ymf1_arg) + C * T.sqr(mf1mod.W2).mean()
mfn_cost = - log_p_yb( ymfn_arg) + C * T.sqr(mfnmod.W2).mean()

updates = {}

alpha = T.scalar()
alpha.tag.test_value = 1e-4

for cost, params in [ (mf1_cost, mf1mod.params()),
                      (mfn_cost, mfnmod.params()) ]:
    for param in params:
        updates[param] = param - alpha * T.grad(cost,param)

from theano import function

func = function([idx,alpha],[mf1_cost,mfn_cost],updates = updates)

dataset = MNIST(which_set = 'test', one_hot = True)

Xt = sharedX(dataset.X)
yt = sharedX(dataset.y)

mf1yt = mf1mod.mf1y(Xt)
mfnyt = mfnmod.mfny(Xt)

ytl = T.argmax(yt,axis=1)

mf1acc = 1.-T.neq(ytl , T.argmax(mf1yt,axis=1)).mean()
mfnacc = 1.-T.neq(ytl , T.argmax(mfnyt,axis=1)).mean()

accs = function([],[mf1acc,mfnacc])


alpha = lr
while True:
    print 'test acc: ',accs()
    print 'doing an epoch'
    for i in xrange(60000/batch_size):
        mf1_cost, mfn_cost = func(i,alpha)
        #print mf1_cost, mfn_cost
