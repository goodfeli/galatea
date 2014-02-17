"""
.. todo::

    WRITEME
"""
__authors__ = 'Ian Goodfellow'
__copyright__ = "Copyright 2013, Universite de Montreal"

import warnings

from theano.compat.python2x import OrderedDict
from theano import config
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import tensor as T

from pylearn2.costs.cost import DefaultDataSpecsMixin, Cost
from pylearn2.models.maxout import Maxout as BaseMaxout
from pylearn2.models.mlp import MLP as BaseMLP
from pylearn2.utils import safe_zip


class Dropout(DefaultDataSpecsMixin, Cost):
    """
    Implements the dropout training technique described in
    "Improving neural networks by preventing co-adaptation of feature
    detectors"
    Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever,
    Ruslan R. Salakhutdinov
    arXiv 2012

    This paper suggests including each unit with probability p during training,
    then multiplying the outgoing weights by p at the end of training.
    We instead include each unit with probability p and divide its
    state by p during training. Note that this means the initial weights should
    be multiplied by p relative to Hinton's.
    The SGD learning rate on the weights should also be scaled by p^2 (use
    W_lr_scale rather than adjusting the global learning rate, because the
    learning rate on the biases should not be adjusted).
    """

    supervised = True

    def __init__(self, default_input_include_prob=.5, input_include_probs=None,
            default_input_scale=2., input_scales=None, per_example=True):
        """
        .. todo::

            WRITEME properly

        During training, each input to each layer is randomly included or excluded
        for each example. The probability of inclusion is independent for each input
        and each example. Each layer uses "default_input_include_prob" unless that
        layer's name appears as a key in input_include_probs, in which case the input
        inclusion probability is given by the corresponding value.

        Each feature is also multiplied by a scale factor. The scale factor for each
        layer's input scale is determined by the same scheme as the input probabilities.
        """

        if input_include_probs is None:
            input_include_probs = {}

        if input_scales is None:
            input_scales = {}

        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, ** kwargs):
        """
        .. todo::

            WRITEME
        """
        space, sources = self.get_data_specs(model)
        space.validate(data)
        (X, Y) = data
        Y_hat = model.dropout_fprop(
            X,
            default_input_include_prob=self.default_input_include_prob,
            input_include_probs=self.input_include_probs,
            default_input_scale=self.default_input_scale,
            input_scales=self.input_scales,
            per_example=self.per_example
        )
        return model.cost(Y, Y_hat)

    def get_gradients(self, model, data, ** kwargs):

        space, sources = self.get_data_specs(model)
        space.validate(data)
        (X, Y) = data
        Y_hat, dynamic_scale = model.rescale_dropout_fprop(
            X,
            default_input_include_prob=self.default_input_include_prob,
            input_include_probs=self.input_include_probs,
            default_input_scale=self.default_input_scale,
            input_scales=self.input_scales,
            per_example=self.per_example
        )

        expr =  model.cost(Y, Y_hat)

        params = model.get_params()
        grads = OrderedDict(safe_zip(params, T.grad(expr, params)))

        for key in grads:
            if key in dynamic_scale:
                grads[key] = grads[key] * dynamic_scale[key]

        return grads, OrderedDict()

class MLP(BaseMLP):


    def rescale_dropout_fprop(self, state_below, default_input_include_prob=0.5,
                      input_include_probs=None, default_input_scale=2.,
                      input_scales=None, per_example=True):
        """
        Returns the output of the MLP, when applying dropout to the input and
        intermediate layers. Each input to each layer is randomly included or
        excluded for each example. The probability of inclusion is independent
        for each input and each example. Each layer uses
        `default_input_include_prob` unless that layer's name appears as a key
        in input_include_probs, in which case the input inclusion probability
        is given by the corresponding value.

        Each feature is also multiplied by a scale factor. The scale factor for
        each layer's input scale is determined by the same scheme as the input
        probabilities.

        Parameters
        ----------
        state_below : WRITEME
            The input to the MLP
        default_input_include_prob : WRITEME
        input_include_probs : WRITEME
        default_input_scale : WRITEME
        input_scales : WRITEME
        per_example : bool, optional
            Sample a different mask value for every example in a batch.
            Defaults to `True`. If `False`, sample one mask per mini-batch.
        """

        warnings.warn("dropout doesn't use fixed_var_descr so it won't work "
                      "with algorithms that make more than one theano "
                      "function call per batch, such as BGD. Implementing "
                      "fixed_var descr could increase the memory usage "
                      "though.")

        if input_include_probs is None:
            input_include_probs = {}

        if input_scales is None:
            input_scales = {}

        self._validate_layer_names(list(input_include_probs.keys()))
        self._validate_layer_names(list(input_scales.keys()))

        theano_rng = MRG_RandomStreams(max(self.rng.randint(2 ** 15), 1))

        dynamic_scale = OrderedDict()

        for layer in self.layers:
            layer_name = layer.layer_name

            if layer_name in input_include_probs:
                include_prob = input_include_probs[layer_name]
            else:
                include_prob = default_input_include_prob

            if layer_name in input_scales:
                scale = input_scales[layer_name]
            else:
                scale = default_input_scale

            state_below = self.apply_dropout(
                state=state_below,
                include_prob=include_prob,
                theano_rng=theano_rng,
                scale=scale,
                mask_value=layer.dropout_input_mask_value,
                input_space=layer.get_input_space(),
                per_example=per_example
            )
            if hasattr(layer, 'dynamic_scale'):
                dynamic_scale.update(layer.dynamic_scale(state_below))
            else:
                print 'skipping', layer.layer_name
            state_below = layer.fprop(state_below)

        return state_below, dynamic_scale


class Maxout(BaseMaxout):

    def dynamic_scale(self, state_below):
        """
        .. todo::

            WRITEME
        """

        self.input_space.validate(state_below)

        if self.requires_reformat:
            state_below = self.input_space.format_as(state_below, self.desired_space)

        z = self.transformer.lmul(state_below) + self.b

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if not hasattr(self, 'pool_stride'):
            self.pool_stride = self.pool_size

        if self.randomize_pools:
            z = T.dot(z, self.permute)

        if not hasattr(self, 'min_zero'):
            self.min_zero = False

        if self.min_zero:
            p = 0.
        else:
            p = None

        last_start = self.detector_layer_dim  - self.pool_size
        for i in xrange(self.pool_size):
            cur = z[:,i:last_start+i+1:self.pool_stride]
            if p is None:
                p = cur
            else:
                p = T.maximum(cur, p)

        cost = p.sum()

        mask = T.grad(cost, z)

        counts = mask.sum(axis=0)


        reweight = T.cast(mask.shape[0], config.floatX) / T.clip(counts, 1.0, 1e6)

        reweight = Print('reweight', attrs=['min', 'mean', 'max'])(reweight)

        params = self.get_params()
        rval = OrderedDict()

        for param in params:
            rval[param] = reweight

        return rval


