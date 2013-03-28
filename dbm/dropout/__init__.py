from pylearn2.models.model import Model
from pylearn2 import utils
from pylearn2.costs.cost import FixedVarDescr
import theano
from pylearn2.space import Conv2DSpace
from pylearn2.space import VectorSpace
from pylearn2.space import CompositeSpace
from pylearn2.utils import sharedX
from pylearn2.linear.conv2d import make_random_conv2D
from pylearn2.linear.conv2d import make_sparse_random_conv2D
import theano.tensor as T
import numpy as np
from pylearn2.expr.probabilistic_max_pooling import max_pool
from pylearn2.expr.probabilistic_max_pooling import max_pool_b01c
from collections import OrderedDict
from pylearn2.utils import block_gradient
import warnings
from theano import function
from theano.sandbox.rng_mrg import MRG_RandomStreams
import time
from pylearn2.costs.cost import Cost
from pylearn2.utils import safe_zip
from pylearn2.utils import safe_izip
from pylearn2.utils import _ElemwiseNoGradient
from pylearn2.utils import safe_union
from theano import config
io = None
from pylearn2.train_extensions import TrainExtension
from pylearn2.models.dbm import block
from pylearn2.models.dbm import BinaryVectorMaxPool
from pylearn2.models.dbm import DBM
from pylearn2.models.dbm import flatten
from pylearn2.models.dbm import HiddenLayer
from pylearn2.models.dbm import VisibleLayer
from pylearn2.models.dbm import InferenceProcedure
from pylearn2.models.dbm import Layer
from pylearn2.models.dbm import WeightDoubling
from pylearn2.models import dbm
from theano.gof.op import get_debug_values
from theano import printing


class UnrollUntie(Model):
    """
    Copied from galatea.dbm.inpaint.MLP_Wrapper to avoid making the code
    too complicated. This version is to support dropout.
    """

    def __init__(self, super_dbm, niter, post_scale = 0.5, input_include_prob = .5):
        self.__dict__.update(locals())
        del self.self
        self.input_space = super_dbm.get_input_space()
        self.output_space = super_dbm.get_output_space()

        self.theano_rng = MRG_RandomStreams(2013+1+27)

        h, g, y = super_dbm.hidden_layers
        vishid = h.get_weights() / post_scale
        biashid = h.get_biases()
        hidpen = g.get_weights() / post_scale
        penhid = g.get_weights().T / post_scale
        biaspen = g.get_biases()
        penlab = y.get_weights() / post_scale
        labpen = y.get_weights().T / post_scale
        biaslab = y.get_biases()

        param_names = ['vishid', 'biashid', 'hidpen', 'penhid', 'biaspen', 'penlab', 'labpen', 'biaslab']
        for name in param_names:
            sh = [ sharedX(locals()[name]) for i in xrange(niter) ]
            setattr(self, name, sh)
        self.penhid[0] = None
        self.labpen[0] = None
        self._params = []
        for name in param_names:
            self._params.extend([elem for elem in getattr(self, name) if elem is not None])
        self.hidden_layers = super_dbm.hidden_layers

    def get_weights(self):
        print 'which iteration? (0-%d)\n' % (len(self.vishid) - 1)
        x = raw_input()
        i = int(x)
        return self.vishid[i].get_value()

    def set_batch_size(self, batch_size):
        self.force_batch_size = batch_size

    def mf(self, V, dropout=False, niter=None, block_grad=None):
        assert niter is None
        assert block_grad is None

        if not dropout:
            weight_names = ['vishid', 'hidpen', 'penhid', 'penlab', 'labpen']
            for name in weight_names:
                current = getattr(self, name)
                setattr(self, 'tmp_'+name, current)
                def scale(x):
                    if x is None:
                        return x
                    return self.post_scale * x
                setattr(self, name, [scale(W) for W in current])

        V = self.apply_dropout(dropout, V, include_prob=self.input_include_prob)

        H1 = T.nnet.sigmoid(T.dot(V, 2. * self.vishid[0]) + self.biashid[0])
        H1 = self.apply_dropout(dropout, H1)
        H2 = T.nnet.sigmoid(T.dot(H1, 2 * self.hidpen[0]) + self.biaspen[0])
        H2 = self.apply_dropout(dropout, H2)
        Y = T.nnet.softmax(T.dot(H2, self.penlab[0]) + self.biaslab[0])
        clean_Y = Y
        Y = self.apply_dropout(dropout, Y, one_hot=True)

        for i in xrange(1, self.niter):
            H1 = T.nnet.sigmoid(T.dot(V, self.vishid[i])+T.dot(H2, self.penhid[i])+self.biashid[i])
            H1 = self.apply_dropout(dropout, H1)
            Y = T.nnet.softmax(T.dot(H2, self.penlab[i]) + self.biaslab[i])
            clean_Y = Y
            Y = self.apply_dropout(dropout, Y, one_hot=True)
            H2 = T.nnet.sigmoid(T.dot(H1, self.hidpen[i]) + T.dot(Y, self.labpen[i]) + self.biaspen[0])
            H2 = self.apply_dropout(dropout, H2)

        if not dropout:
            for name in weight_names:
                setattr(self, name, getattr(self, 'tmp_'+name))

        return [H1, H2, clean_Y]

    def apply_dropout(self, dropout, X, one_hot=False, include_prob=.5):
        if dropout:
            if one_hot:
                X = X * (self.theano_rng.binomial(p=include_prob, n=1, size=(X.shape[0],), dtype=X.dtype).dimshuffle(0, 'x'))
            else:
                X = X * self.theano_rng.binomial(p=include_prob, n=1, size=X.shape, dtype=X.dtype)
        return X

class FeatureMLP(Model):
    """
    Copied from galatea.dbm.inpaint.MLP_Wrapper to avoid making the code
    too complicated. This version is to support dropout.
    """

    def __init__(self, super_dbm, feature_niter, post_scale = 0.5, input_include_prob = .5,
            remove_y = False):
        self.__dict__.update(locals())
        del self.self
        self.input_space = super_dbm.get_input_space()
        self.output_space = super_dbm.get_output_space()

        self.theano_rng = MRG_RandomStreams(2013+1+27)

        h, g, y = super_dbm.hidden_layers
        vishid = h.get_weights()
        biashid = h.get_biases()
        hidpen = g.get_weights()
        penhid = g.get_weights().T
        biaspen = g.get_biases()
        penlab = y.get_weights()
        labpen = y.get_weights().T
        biaslab = y.get_biases()

        param_names = ['vishid', 'biashid', 'hidpen', 'penhid', 'biaspen', 'penlab', 'labpen', 'biaslab']
        self._params = []
        for name in param_names:
            val = locals()[name]
            scaled_val = val
            if val.ndim == 2:
                scaled_val = val / post_scale
            param = sharedX(scaled_val)
            setattr(self, name, param)
            self._params.append(param)
            fixed = sharedX(val)
            setattr(self, 'feature_'+name, fixed)
        self.hidden_layers = super_dbm.hidden_layers

    def get_weights(self):
        return self.vishid.get_value()

    def set_batch_size(self, batch_size):
        self.force_batch_size = batch_size

    def mf(self, V, dropout=False, niter=None, block_grad=None):
        assert niter is None
        assert block_grad is None

        H1 = T.nnet.sigmoid(T.dot(V, 2. * self.feature_vishid) + self.feature_biashid)
        H2 = T.nnet.sigmoid(T.dot(H1, 2 * self.feature_hidpen) + self.feature_biaspen)
        for i in xrange(1, self.feature_niter):
            H1 = T.nnet.sigmoid(T.dot(V, self.feature_vishid)+T.dot(H2, self.feature_penhid)+self.feature_biashid)
            Y = T.nnet.softmax(T.dot(H2, self.feature_penlab) + self.feature_biaslab)
            if self.remove_y:
                Y = T.zeros_like(Y)
            H2 = T.nnet.sigmoid(T.dot(H1, self.feature_hidpen) + T.dot(Y, self.feature_labpen) + self.feature_biaspen)

        if not dropout:
            weight_names = ['vishid', 'hidpen', 'penhid', 'penlab', 'labpen']
            for name in weight_names:
                current = getattr(self, name)
                setattr(self, 'tmp_'+name, current)
                def scale(x):
                    if x is None:
                        return x
                    return self.post_scale * x
                setattr(self, name, scale(current))

        V = self.apply_dropout(dropout, V, include_prob=self.input_include_prob)
        H2 = self.apply_dropout(dropout, H2)
        if not self.remove_y:
            Y = self.apply_dropout(dropout, Y, one_hot=True)

        H1 = T.nnet.sigmoid(T.dot(V, self.vishid)+T.dot(H2, self.penhid)+self.biashid)
        H1 = self.apply_dropout(dropout, H1)
        if not self.remove_y:
            Y = T.nnet.softmax(T.dot(H2, self.penlab) + self.biaslab)
            Y = self.apply_dropout(dropout, Y, one_hot=True)
        H2 = T.nnet.sigmoid(T.dot(H1, self.hidpen) + T.dot(Y, self.labpen) + self.biaspen)
        Y = T.nnet.softmax(T.dot(H2, self.penlab) + self.biaslab)

        if not dropout:
            for name in weight_names:
                setattr(self, name, getattr(self, 'tmp_'+name))

        return [H1, H2, Y]

    def apply_dropout(self, dropout, X, one_hot=False, include_prob=.5):
        if dropout:
            if one_hot:
                X = X * (self.theano_rng.binomial(p=include_prob, n=1, size=(X.shape[0],), dtype=X.dtype).dimshuffle(0, 'x'))
            else:
                X = X * self.theano_rng.binomial(p=include_prob, n=1, size=X.shape, dtype=X.dtype)
        return X

class DropoutDBM_ConditionalNLL(Cost):

    supervised = True

    def __init__(self):
        self.__dict__.update(locals())

    def Y_hat(self, model, X, dropout=False):
        assert isinstance(model.hidden_layers[-1], dbm.Softmax)
        Y_hat = model.mf(X, dropout=dropout)[-1]
        Y_hat.name = 'Y_hat'

        return Y_hat

    def __call__(self, model, X, Y, dropout = True, **kwargs):
        """ Returns - log P( Y | X) / m
            where Y is a matrix of one-hot labels,
            one label per row
            X is a batch of examples, X[i,:] being an example
            (but not necessarily a row, ie, could be an image)
            P is given by the model (see the __init__ docstring
            for details)
            m is the number of examples
        """

        assert 'niter' not in kwargs

        Y_hat = self.Y_hat(model, X, dropout=dropout)
        assert Y_hat.ndim == 2
        assert Y.ndim == 2

        # Pull out the argument to the softmax
        assert hasattr(Y_hat, 'owner')
        assert Y_hat.owner is not None
        assert isinstance(Y_hat.owner.op, T.nnet.Softmax)
        arg ,= Y_hat.owner.inputs
        arg.name = 'arg'

        arg = arg - arg.max(axis=1).dimshuffle(0,'x')
        arg.name = 'safe_arg'

        unnormalized = T.exp(arg)
        unnormalized.name = 'unnormalized'

        Z = unnormalized.sum(axis=1)
        Z.name = 'Z'

        log_ymf = arg - T.log(Z).dimshuffle(0,'x')

        log_ymf.name = 'log_ymf'

        example_costs =  Y * log_ymf
        example_costs.name = 'example_costs'

        return - example_costs.mean()

    def get_monitoring_channels(self, model, X, Y, **kwargs):

        y = T.argmax(Y, axis=1)

        def acc(y, dropout):
            Y_hat = self.Y_hat(model, X, dropout=dropout)
            y = T.cast(y, Y_hat.dtype)

            argmax = T.argmax(Y_hat,axis=1)
            if argmax.dtype != Y_hat.dtype:
                argmax = T.cast(argmax, Y_hat.dtype)
            neq = T.neq(y , argmax).mean()
            if neq.dtype != Y_hat.dtype:
                neq = T.cast(neq, Y_hat.dtype)
            acc = 1.- neq

            assert acc.dtype == Y_hat.dtype
            return acc

        rval = { 'misclass' : 1. - acc(y, dropout=False),
                 'dropout_misclass' : 1. - acc(y, dropout=True),
                 'nll' : self(model, X, Y, dropout=False, **kwargs) }

        return rval
