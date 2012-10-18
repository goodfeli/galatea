from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from theano.sandbox.rng_mrg import MRG_RandomStreams
from pylearn2.utils import sharedX
import numpy as np
import theano.tensor as T
from pylearn2.costs.cost import Cost
from theano.printing import Print

class SimpleModel(Model):

    def __init__(self, nvis, num_hid, num_class):
        self.__dict__.update(locals())
        del self.self

        self.input_space = VectorSpace(nvis)
        self.output_space = VectorSpace(num_class)
        self.theano_rng = MRG_RandomStreams(2012 + 10 + 16)
        rng = np.random.RandomState([16,10,2012])

        self.W = sharedX(rng.uniform(-.05,.05,(nvis, num_hid)))
        self.hb = sharedX(np.zeros((num_hid,)) - 1.)
        self.V = sharedX(rng.uniform(-.05,.05,(num_hid, num_class)))
        self.cb = sharedX(np.zeros((num_class,)))

        self._params = [self.W, self.hb, self.V, self.cb ]

    def get_weights(self):
        return self.W.get_value()

    def get_weights_format(self):
        return ('v','h')

    def emit(self, X):

        Z = T.dot(X, self.W) + self.hb
        exp_H = T.nnet.sigmoid(Z)
        H = self.theano_rng.binomial(p = exp_H, n = 1, size = exp_H.shape, dtype = exp_H.dtype)

        Zc = T.dot(H, self.V) + self.cb

        return exp_H, H, Zc

def log_prob(Z):
    Z = Z - Z.max(axis=1).dimshuffle(0, 'x')

    rval =  Z - T.log(T.exp(Z).sum(axis=1)).dimshuffle(0,'x')

    #rval = Print('log_prob', attrs = ['min'])(rval)

    return rval

def log_prob_of(Y, Z):
    return (Y * log_prob(Z)).sum(axis=1)

def prob_of(Y,Z):
    return (Y * T.nnet.softmax(Z)).sum(axis=1)

class SamplingCost(Cost):
    supervised = True

    def __init__(self, weight_decay_1=0., weight_decay_2=0.):
        self.__dict__.update(locals())
        del self.self

    def batch_loss(self, Y, Z):
        return - log_prob_of(Y, Z)

    def __call__(self, model, X, Y):
        assert type(model) is SimpleModel # yes, I did not mean to use isinstance
        _, __, Z = model.emit(X)
        return self.batch_loss(Y, Z).mean()

    def get_gradients(self, model, X, Y):

        obj = self(model, X, Y)
        exp_H, H, Z = model.emit(X)
        batch_loss = self.batch_loss(Y, Z)

        rval = {}

        for param in [model.V, model.cb]:
            rval[param] = T.grad(obj, param)

        grad_Z = batch_loss.dimshuffle(0, 'x') * \
                        (H-exp_H)

        rval[model.hb] = grad_Z.mean(axis=0)
        rval[model.W] = T.dot(X.T,grad_Z)/ T.cast(X.shape[0], 'float32')


        rval[model.W] += self.weight_decay_1 * model.W
        rval[model.V] += self.weight_decay_2 * model.V

        return rval, {}

    def get_monitoring_channels(self, model, X, Y):
        _, __, Z = model.emit(X)

        return { 'acc' : T.cast(T.eq(T.argmax(Z,axis=1),T.argmax(Y,axis=1)).mean(), 'float32') }

class SampledClassModel(SimpleModel):


    def emit(self, X):

        Z = T.dot(X, self.W) + self.hb
        exp_H = T.nnet.sigmoid(Z)
        H = self.theano_rng.binomial(p = exp_H, n = 1, size = exp_H.shape, dtype = exp_H.dtype)

        Zc = T.dot(H, self.V) + self.cb
        #exp_C = T.nnet.softmax(Zc)
        #C = self.theano_rng.multinomial(pvals = exp_C, dtype = exp_C.dtype)
        exp_C = T.nnet.sigmoid(Zc)
        C = self.theano_rng.binomial(p = exp_C, n=1, size = exp_C.shape, dtype=exp_C.dtype)

        return exp_H, H, exp_C, C

class ZeroOneLoss(SamplingCost):
    supervised = True

    #def __init__(self, * args, ** kwargs):
    #    super(ZeroOneLoss, self).__init__(*args, **kwargs)

    def loss_matrix(self, Y, C):
        return abs(Y-C)

    def batch_loss(self, Y, C):
        return self.loss_matrix(Y, C).sum(axis=1)

    def __call__(self, model, X, Y):
        assert isinstance(model, SampledClassModel)
        _, __, ___, C = model.emit(X)

        return self.batch_loss(Y, C).mean()

    def get_gradients(self, model, X, Y):

        obj = self(model, X, Y)
        exp_H, H, exp_C, C = model.emit(X)
        loss_matrix = self.loss_matrix(Y, C)
        #offsets = loss_matrix.mean(axis=0)
        #loss_matrix = loss_matrix - offsets
        batch_loss = loss_matrix.sum(axis=1).dimshuffle(0,'x')

        rval = {}

        grad_Z = batch_loss * (H - exp_H)

        m = T.cast(X.shape[0], 'float32')

        rval[model.hb] = grad_Z.mean(axis=0)
        rval[model.W] = T.dot(X.T,grad_Z) / m

        diff = C - exp_C
        grad_Z = loss_matrix * diff

        rval[model.cb] = grad_Z.mean(axis=0)
        rval[model.V] = T.dot(H.T,grad_Z)/ m

        rval[model.W] += self.weight_decay_1 * model.W
        rval[model.V] += self.weight_decay_2 * model.V

        return rval, {}

    def get_monitoring_channels(self, model, X, Y):

        exp_H, H, exp_C, C = model.emit(X)

        return { 'exp_C.mean' : exp_C.mean() }




class SamplingCost2(Cost):
    supervised = True

    def __init__(self, weight_decay_1=0., weight_decay_2=0.):
        self.__dict__.update(locals())
        del self.self

    def batch_loss(self, Y, Z):
        return 1. - prob_of(Y, Z)

    def __call__(self, model, X, Y):
        assert type(model) is SimpleModel # yes, I did not mean to use isinstance
        _, __, Z = model.emit(X)
        return self.batch_loss(Y, Z).mean()

    def get_gradients(self, model, X, Y):

        obj = self(model, X, Y)
        exp_H, H, Z = model.emit(X)
        batch_loss = self.batch_loss(Y, Z)

        rval = {}

        for param in [model.V, model.cb]:
            rval[param] = T.grad(obj, param)

        grad_Z = batch_loss.dimshuffle(0, 'x') * \
                        (H-exp_H)

        rval[model.hb] = grad_Z.mean(axis=0)
        rval[model.W] = T.dot(X.T,grad_Z)/ T.cast(X.shape[0], 'float32')


        rval[model.W] += self.weight_decay_1 * model.W
        rval[model.V] += self.weight_decay_2 * model.V

        return rval, {}

    def get_monitoring_channels(self, model, X, Y):
        _, __, Z = model.emit(X)

        return { 'acc' : T.cast(T.eq(T.argmax(Z,axis=1),T.argmax(Y,axis=1)).mean(), 'float32') }


class SimpleModel2(Model):

    def __init__(self, nvis, num_hid, num_hid_2, num_class):
        self.__dict__.update(locals())
        del self.self

        self.input_space = VectorSpace(nvis)
        self.output_space = VectorSpace(num_class)
        self.theano_rng = MRG_RandomStreams(2012 + 10 + 16)
        rng = np.random.RandomState([16,10,2012])

        self.W = sharedX(rng.uniform(-.05,.05,(nvis, num_hid)))
        self.hb = sharedX(np.zeros((num_hid,)) - 1.)
        self.V = sharedX(rng.uniform(-.05,.05,(num_hid, num_hid_2)))
        self.gb = sharedX(np.zeros((num_hid_2,)) - 1.)
        self.V2 = sharedX(rng.uniform(-.05,.05,(num_hid_2, num_class)))
        self.cb = sharedX(np.zeros((num_class,)))

        self._params = [self.W, self.hb, self.V, self.V2, self.gb, self.cb ]

    def get_weights(self):
        return self.W.get_value()

    def get_weights_format(self):
        return ('v','h')

    def emit(self, X):

        Z = T.dot(X, self.W) + self.hb
        exp_H = T.nnet.sigmoid(Z)
        H = self.theano_rng.binomial(p = exp_H, n = 1, size = exp_H.shape, dtype = exp_H.dtype)

        Z = T.dot(H, self.V) + self.gb
        exp_G = T.nnet.sigmoid(Z)
        G = self.theano_rng.binomial(p = exp_G, n = 1, size = exp_G.shape, dtype = exp_G.dtype)

        Zc = T.dot(H, self.V2) + self.cb

        return exp_H, H, exp_G, G, Zc

    def add_polyak_channels(self, params, d):
        X = T.matrix()
        Y = T.matrix()

        W = params[self.W]
        hb = params[self.hb]
        V = params[self.V]
        gb = params[self.gb]
        V2 = params[self.V2]
        cb = params[self.cb]

        Z = T.dot(X, W) + hb
        exp_H = T.nnet.sigmoid(Z)
        H = self.theano_rng.binomial(p = exp_H, n = 1, size = exp_H.shape, dtype = exp_H.dtype)

        Z = T.dot(H, V) + gb
        exp_G = T.nnet.sigmoid(Z)
        G = self.theano_rng.binomial(p = exp_G, n = 1, size = exp_G.shape, dtype = exp_G.dtype)

        Z = T.dot(H, V2) + cb

        polyak_acc = T.cast(T.eq(T.argmax(Z,axis=1),T.argmax(Y,axis=1)).mean(), 'float32')

        for n in d:
            ds = d[n]
            name = n+'_polyak_acc'
            self.monitor.add_channel(name, (X, Y), polyak_acc, ds)

class SamplingCost3(Cost):
    supervised = True

    def __init__(self, weight_decay_1=0., weight_decay_2=0.,
            weight_decay_3=0.):
        self.__dict__.update(locals())
        del self.self

    def batch_loss(self, Y, Z):
        return - log_prob_of(Y, Z)

    def __call__(self, model, X, Y):
        assert type(model) is SimpleModel2 # yes, I did not mean to use isinstance
        eH, H, eG, G, Z = model.emit(X)
        return self.batch_loss(Y, Z).mean()

    def get_gradients(self, model, X, Y):

        obj = self(model, X, Y)
        exp_H, H, exp_G, G, Z = model.emit(X)
        batch_loss = self.batch_loss(Y, Z)
        mn = sharedX(0.)
        # we want to subtract a constant that's close to the
        # true expected value of the cost. this reduces variance.
        # we can't use the current samples to estimate this constant
        # though. so we use earlier batch's samples
        batch_loss = batch_loss - mn
        alpha = .01
        updates = { mn : alpha * batch_loss.mean() + (1.-alpha)*mn }

        rval = {}

        for param in [model.V2, model.cb]:
            rval[param] = T.grad(obj, param)

        grad_Z = batch_loss.dimshuffle(0, 'x') * \
                        (H-exp_H)

        rval[model.hb] = grad_Z.mean(axis=0)
        rval[model.W] = T.dot(X.T,grad_Z)/ T.cast(X.shape[0], 'float32')

        grad_Z = batch_loss.dimshuffle(0, 'x') * \
                        (G-exp_G)

        rval[model.gb] = grad_Z.mean(axis=0)
        rval[model.V] = T.dot(H.T,grad_Z)/ T.cast(H.shape[0], 'float32')


        rval[model.W] += self.weight_decay_1 * model.W
        rval[model.V] += self.weight_decay_2 * model.V
        rval[model.V2] += self.weight_decay_3 * model.V2

        return rval, updates

    def get_monitoring_channels(self, model, X, Y):
        _, __, ___, ____, Z = model.emit(X)

        return { 'acc' : T.cast(T.eq(T.argmax(Z,axis=1),T.argmax(Y,axis=1)).mean(), 'float32') }
