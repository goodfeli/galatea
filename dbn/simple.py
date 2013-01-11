import numpy as np

from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from pylearn2.utils import block_gradient
from pylearn2.utils import sharedX
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import tensor as T

def sigmoid_prob(z, x):
    assert z.ndim == 2
    assert x.ndim == 2
    batch_prob = -x * T.nnet.softplus(-z) - (1 - x) * T.nnet.softplus(z)
    return batch_prob.sum(axis=1).mean(axis=0)

def softmax_prob(z, x):
    assert z.ndim == 2
    assert x.ndim == 2
    z = z - z.max(axis=1).dimshuffle(0, 'x')
    log_prob = z - T.exp(z).sum(axis=1).dimshuffle(0, 'x')
    log_prob_of = (x * log_prob).sum(axis=1)
    return log_prob_of.mean(axis=0)


class SimpleDBN(Model):

    def __init__(self):
        super(SimpleDBN, self).__init__()

        self.n_classes = 10
        self.nvis = 784
        self.nhid1 = 500
        self.nhid2 = 1000
        self.force_batch_size = 1

        self.input_space = VectorSpace(self.nvis)
        self.output_space = VectorSpace(self.n_classes)
        self.rng = np.random.RandomState([2012, 12, 18])

        self.rh1b1 = self.make_biases(self.nhid1, -1.)
        self.rh1b2 = self.make_biases(self.nhid1, -1.)
        self.rh1w1 = self.make_weights(self.nvis, self.nhid1, .05)
        self.rh1w2 = sharedX(self.rh1w1.get_value())
        self.rh1w3 = self.make_weights(self.nhid1, self.nhid1, .05)
        self.rh2b = self.make_biases(self.nhid2, -1.)
        self.rh2w = self.make_weights(self.nhid1, self.nhid2, .05)
        self.ryb = self.make_biases(self.n_classes, -1.)
        self.ryw = self.make_weights(self.nhid2, self.n_classes, .05)

        self.gvb = self.make_biases(self.nvis, -1.)
        self.gvw = sharedX(self.rh1w1.get_value().T)
        self.gh1b = self.make_biases(self.nhid1, -1.)
        self.gh1w = sharedX(self.rh2w.get_value().T)
        self.gh2b = self.make_biases(self.nhid2, -1.)
        self.gh2w = sharedX(self.ryw.get_value().T)
        self.gyb = self.make_biases(self.n_classes, -1.)

        self._params = []
        for elem in dir(self):
            try:
                elem = getattr(self, elem)
            except:
                continue
            if hasattr(elem, 'get_value'):
                self._params.append(elem)

    def set_batch_size(self, batch_size):
        self.force_batch_size = batch_size
        self.batch_size = batch_size

    def get_monitoring_channels(self, X, Y, **kwargs):

        theano_rng = MRG_RandomStreams(2012 + 12 + 19)
        # Explanation of reality
        zh1, rh1 = self.infer_h1(X)
        rh1 = block_gradient(rh1)
        zh2 = T.dot(rh1, self.rh2w) + self.rh2b
        rh2 = theano_rng.binomial(p = T.nnet.sigmoid(zh2), size = zh2.shape, dtype='float32')
        rh2 = block_gradient(rh2)
        y = T.dot(rh2, self.ryw) + self.ryb

        err = T.neq(T.argmax(y, axis=1), T.argmax(Y, axis=1))
        assert err.ndim == 1

        return { 'misclass' : err.astype('float32').mean() }

    def infer_h1(self, x):

        first = T.nnet.sigmoid(T.dot(x, self.rh1w1) + self.rh1b1)
        z = T.dot(x, self.rh1w2) + T.dot(first, self.rh1w3) + self.rh1b2

        return z, T.nnet.sigmoid(z)

    def get_cost(self, X, Y, **kwargs):

        # Dream
        theano_rng = MRG_RandomStreams(2012 + 12 + 18)
        exp_y = T.nnet.softmax(T.alloc(0., self.batch_size, self.n_classes) + self.gyb)
        dy = theano_rng.multinomial(pvals = exp_y, dtype='float32')
        dy = block_gradient(dy)
        exp_h2 = T.nnet.sigmoid(T.dot(dy, self.gh2w) + self.gh2b)
        dh2 = theano_rng.binomial(p = exp_h2, size = exp_h2.shape, dtype='float32')
        dh2 = block_gradient(dh2)
        exp_h1 = T.nnet.sigmoid(T.dot(dh2, self.gh1w) + self.gh1b)
        dh1 = theano_rng.binomial(p = exp_h1, size = exp_h1.shape, dtype='float32')
        dh1 = block_gradient(dh1)
        exp_v = T.nnet.sigmoid(T.dot(dh1, self.gvw) + self.gvb)
        dv = theano_rng.binomial(p = exp_v, size = exp_v.shape, dtype='float32')
        dv = block_gradient(dv)

        # Explanation of dream
        zh1, rh1 = self.infer_h1(dv)
        zh2 = T.dot(rh1, self.rh2w) + self.rh2b
        rh2 = T.nnet.sigmoid(zh2)
        zy = T.dot(rh2, self.ryw) + self.ryb

        # Probability of dream
        dream_prob = sigmoid_prob(zh1, dh1) + sigmoid_prob(zh2, dh2) + softmax_prob(zy, dy)

        # Explanation of reality
        zh1, rh1 = self.infer_h1(X)
        rh1 = block_gradient(rh1)
        zh2 = T.dot(rh1, self.rh2w) + self.rh2b
        rh2 = theano_rng.binomial(p = T.nnet.sigmoid(zh2), size = zh2.shape, dtype='float32')
        rh2 = block_gradient(rh2)

        # Probability of reality
        real_prob = softmax_prob(T.alloc(0., self.batch_size, self.n_classes) + self.gyb, Y) + \
                sigmoid_prob(T.dot(Y, self.gh2w) + self.gh2b, rh2) + \
                sigmoid_prob(T.dot(rh2, self.gh1w) + self.gh1b, rh1) + \
                sigmoid_prob(T.dot(rh1, self.gvw) + self.gvb, X)

        return - dream_prob - real_prob + .0001 * (
            T.sqr(self.gvw).sum() + T.sqr(self.gh1w).sum() + \
                    T.sqr(self.gh2w).sum()
                )

    def make_weights(self, r, c, irange):

        rval = np.zeros((r,c))
        for i in xrange(15):
            for j in xrange(r):
                rval[j,self.rng.randint(c)] = self.rng.randn()
        return sharedX(rval)

        return sharedX(self.rng.uniform(-irange, irange, (r, c)))

    def make_biases(self, dim, init):
        return sharedX(np.zeros((dim,)) + init)

    def get_weights(self):
        x = raw_input('which weights? (g)enerative or (r1/r2)ecognition? ')
        if x == 'r1':
            return self.rh1w1.get_value()
        elif x == 'r2':
            return self.rh1w2.get_value()
        elif x == 'g':
            return self.gvw.get_value().T
        else:
            assert False

    def get_weights_format(self):
        return ('v', 'h')

