import functools
import theano
import numpy
from theano.compat import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import tensor as T

from pylearn2.space import VectorSpace
from pylearn2.costs.cost import Cost
from pylearn2.costs.cost import DefaultDataSpecsMixin
from pylearn2.expr.nnet import kl
from pylearn2.models.mlp import Layer
from pylearn2.models import Model
from pylearn2.space import CompositeSpace
from pylearn2.utils import block_gradient
from pylearn2.utils import safe_zip
from pylearn2.utils import serial

class AdversaryPair(Model):

    def __init__(self, generator, discriminator, inferer=None,
                 inference_monitoring_batch_size=128):
        Model.__init__(self)
        self.__dict__.update(locals())
        del self.self

    def get_params(self):
        p = self.generator.get_params() + self.discriminator.get_params()
        if self.inferer is not None:
            p += self.inferer.get_params()
        return p

    def get_input_space(self):
        return self.discriminator.get_input_space()

    def get_weights_topo(self):
        return self.discriminator.get_weights_topo()

    def get_weights(self):
        return self.discriminator.get_weights()

    def get_weights_format(self):
        return self.discriminator.get_weights_format()

    def get_weights_view_shape(self):
        return self.discriminator.get_weights_view_shape()

    def get_monitoring_channels(self, data):
        rval = OrderedDict()

        g_ch = self.generator.get_monitoring_channels(data)
        # need to spoof targets: d_ch = self.discriminator.get_monitoring_channels(data)
        d_ch = OrderedDict()

        i_ch = OrderedDict()
        if self.inferer is not None:
            batch_size = self.inference_monitoring_batch_size
            sample, noise = self.generator.sample_and_noise(batch_size)
            i_ch.update(self.inferer.get_monitoring_channels((sample, noise)))

        for key in g_ch:
            rval['gen_' + key] = g_ch[key]
        for key in d_ch:
            rval['dis_' + key] = d_ch[key]
        for key in i_ch:
            rval['inf_' + key] = i_ch[key]
        return rval

    def get_monitoring_data_specs(self):

        space = self.discriminator.get_input_space()
        source = self.discriminator.get_input_source()
        return (space, source)

    def _modify_updates(self, updates):
        self.generator.modify_updates(updates)
        self.discriminator.modify_updates(updates)
        if self.inferer is not None:
            self.inferer.modify_updates(updates)

    def get_lr_scalers(self):

        rval = self.generator.get_lr_scalers()
        rval.update(self.discriminator.get_lr_scalers())
        return rval


class Generator(Model):

    def __init__(self, mlp, monitor_ll = False, ll_n_samples = 100, ll_sigma = 0.2):
        Model.__init__(self)
        self.__dict__.update(locals())
        del self.self
        self.theano_rng = MRG_RandomStreams(2014 * 5 + 27)



    def sample_and_noise(self, num_samples, default_input_include_prob=1., default_input_scale=1.):
        n = self.mlp.get_input_space().get_total_dimension()
        noise = self.theano_rng.normal(size=(num_samples, n), dtype='float32')
        formatted_noise = VectorSpace(n).format_as(noise, self.mlp.get_input_space())
        return self.mlp.dropout_fprop(formatted_noise, default_input_include_prob=default_input_include_prob, default_input_scale=default_input_scale), formatted_noise

    def sample(self, num_samples, default_input_include_prob=1., default_input_scale=1.):
        sample, _ = self.sample_and_noise(num_samples, default_input_include_prob, default_input_scale)
        return sample

    def get_monitoring_channels(self, data):
        if data is None:
            m = 100
        else:
            m = data.shape[0]
        n = self.mlp.get_input_space().get_total_dimension()
        noise = self.theano_rng.normal(size=(m, n), dtype='float32')
        rval = self.mlp.get_monitoring_channels((noise, None))
        if  self.monitor_ll:
            rval['ll'] = T.cast(self.ll(data, self.ll_n_samples, self.ll_sigma),
                                        theano.config.floatX).mean()
            rval['nll'] = -rval['ll']
        return rval

    def get_params(self):
        return self.mlp.get_params()

    def get_output_space(self):
        return self.mlp.get_output_space()

    def ll(self, data, n_samples, sigma):

        samples = self.sample(n_samples)
        parzen = theano_parzen(data, samples, sigma)
        return parzen

    def _modify_updates(self, updates):
        self.mlp.modify_updates(updates)

    def get_lr_scalers(self):
        return self.mlp.get_lr_scalers()



class IntrinsicDropoutGenerator(Generator):
    def __init__(self, default_input_include_prob, default_input_scale, **kwargs):
        super(IntrinsicDropoutGenerator, self).__init__(**kwargs)
        self.__dict__.update(locals())
        del self.self

    def sample(self, num_samples, default_input_include_prob=None, default_input_scale=None):
        # ignores dropout args
        default_input_include_prob = self.default_input_include_prob
        default_input_scale = self.default_input_scale
        # Assumes design matrix
        n = self.mlp.get_input_space().get_total_dimension()
        noise = self.theano_rng.normal(size=(num_samples, n), dtype='float32')
        return self.mlp.dropout_fprop(noise, default_input_include_prob=default_input_include_prob, default_input_scale=default_input_scale)



# Used to be AdversaryCost, but has a bug. Use AdversaryCost2
class BadAdversaryCost(DefaultDataSpecsMixin, Cost):
    """
    """

    # Supplies own labels, don't get them from the dataset
    supervised = False

    def __init__(self, scale_grads=1, target_scale=.1,
            discriminator_default_input_include_prob = 1.,
            discriminator_input_include_probs=None,
            discriminator_default_input_scale=1.,
            discriminator_input_scales=None,
            generator_default_input_include_prob = 1.,
            generator_default_input_scale=1.):
        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, **kwargs):
        """
        The generator and discriminator objectives always cancel out.
        """
        return T.constant(0.)

    def get_gradients(self, model, data, **kwargs):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        assert isinstance(model, AdversaryPair)
        g = model.generator
        d = model.discriminator

        # Note: this assumes data is design matrix
        X = data
        m = X.shape[0]
        y = T.concatenate((T.alloc(1, m, 1), T.alloc(0, m, 1)), axis=0)
        S = g.sample(m, default_input_include_prob=self.generator_default_input_include_prob, default_input_scale=self.generator_default_input_scale)
        discriminator_X = T.concatenate((X, S), axis=0)
        y_hat = d.dropout_fprop(discriminator_X, self.discriminator_default_input_include_prob,
                                     self.discriminator_input_include_probs,
                                     self.discriminator_default_input_scale,
                                     self.discriminator_input_scales)

        obj =  d.layers[-1].cost(y, y_hat)
        g_params = g.get_params()
        d_params = d.get_params()
        for param in g_params:
            assert param not in d_params
        for param in d_params:
            assert param not in g_params
        d_grads = T.grad(obj, d_params)
        g_grads = T.grad(-obj, g_params)

        if self.scale_grads:
            S_grad = T.grad(obj, S)
            scale = T.maximum(1., self.target_scale / T.sqrt(T.sqr(S_grad).sum()))
            g_grads = [g_grad * scale for g_grad in g_grads]

        rval = OrderedDict(safe_zip(d_params + g_params,
            d_grads + g_grads))
        return rval, OrderedDict()

    def get_monitoring_channels(self, model, data, **kwargs):

        rval = OrderedDict()

        m = data.shape[0]

        g = model.generator
        d = model.discriminator

        y_hat = d.fprop(data)

        rval['false_negatives'] = T.cast((y_hat < 0.5).mean(), 'float32')

        samples = g.sample(m)
        y_hat = d.fprop(samples)
        rval['false_positives'] = T.cast((y_hat > 0.5).mean(), 'float32')
        # y = T.alloc(0., m, 1)
        cost = d.cost_from_X((samples, y_hat))
        sample_grad = T.grad(-cost, samples)
        rval['sample_grad_norm'] = T.sqrt(T.sqr(sample_grad).sum())

        return rval


class IndistCost(DefaultDataSpecsMixin, Cost):
    """
    """

    # Supplies own labels, don't get them from the dataset
    supervised = False

    def __init__(self, scale_grads=1, target_scale=.1,
            discriminator_default_input_include_prob = 1.,
            discriminator_input_include_probs=None,
            discriminator_default_input_scale=1.,
            discriminator_input_scales=None,
            generator_default_input_include_prob = 1.,
            generator_default_input_scale=1.):
        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, **kwargs):
        S, discrim_obj, gen_obj = self.foo(model, data)
        return discrim_obj + gen_obj

    def foo(self, model, data):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        assert isinstance(model, AdversaryPair)
        g = model.generator
        d = model.discriminator
        # Note: this assumes data is design matrix
        X = data
        m = X.shape[0]
        y1 = T.alloc(1, m, 1)
        y0 = T.alloc(0, m, 1)
        S = g.sample(m, default_input_include_prob=self.generator_default_input_include_prob, default_input_scale=self.generator_default_input_scale)
        # discriminator_X = T.concatenate((X, S), axis=0)
        y_hat1 = d.dropout_fprop(X, self.discriminator_default_input_include_prob,
                                     self.discriminator_input_include_probs,
                                     self.discriminator_default_input_scale,
                                     self.discriminator_input_scales)
        y_hat0 = d.dropout_fprop(S, self.discriminator_default_input_include_prob,
                                     self.discriminator_input_include_probs,
                                     self.discriminator_default_input_scale,
                                     self.discriminator_input_scales)

        discrim_obj =  0.5 * (d.layers[-1].cost(y1, y_hat1) + d.layers[-1].cost(y0, y_hat0))
        g_params = g.get_params()
        d_params = d.get_params()
        for param in g_params:
            assert param not in d_params
        for param in d_params:
            assert param not in g_params
        # d_grads = T.grad(discrim_obj, d_params)

        # KL(B(p), B(sigmoid(z)))
        # p log (p/sigmoid(z)) + (1 - p) log (1 - p)/sigmoid(-z)
        # p log p - p log sigmoid(z) + (1-p) log (1- p) - (1 - p) log sigmoid(-z)
        # 0.5 log 0.5 - 0.5 log sigmoid(z) + 0.5 log 0.5 - 0.5 log sigmoid(-z)
        # log 0.5 - 0.5 log sigmoid(z) - 0.5 log sigmoid(-z)
        # log 0.5 + 0.5 softplus(-z) + 0.5 softplus(z)

        gen_obj = kl(T.alloc(0.5, m, 1), y_hat0, batch_axis=0).mean() + T.log(0.5)

        return S, discrim_obj, gen_obj

    def get_gradients(self, model, data, **kwargs):
        assert isinstance(model, AdversaryPair)
        g = model.generator
        d = model.discriminator

        S, discrim_obj, gen_obj = self.foo(model, data)

        g_params = g.get_params()
        d_params = d.get_params()
        for param in g_params:
            assert param not in d_params
        for param in d_params:
            assert param not in g_params
        d_grads = T.grad(discrim_obj, d_params)

        g_grads = T.grad(gen_obj, g_params)

        if self.scale_grads:
            S_grad = T.grad(gen_obj, S)
            scale = T.maximum(1., self.target_scale / T.sqrt(T.sqr(S_grad).sum()))
            g_grads = [g_grad * scale for g_grad in g_grads]

        rval = OrderedDict(safe_zip(d_params + g_params,
            d_grads + g_grads))
        return rval, OrderedDict()

    def get_monitoring_channels(self, model, data, **kwargs):

        rval = OrderedDict()

        m = data.shape[0]

        g = model.generator
        d = model.discriminator

        y_hat = d.fprop(data)

        rval['false_negatives'] = T.cast((y_hat < 0.5).mean(), 'float32')

        samples = g.sample(m)
        y_hat = d.fprop(samples)
        rval['false_positives'] = T.cast((y_hat > 0.5).mean(), 'float32')
        # y = T.alloc(0., m, 1)
        cost = d.cost_from_X((samples, y_hat))
        sample_grad = T.grad(-cost, samples)
        rval['sample_grad_norm'] = T.sqrt(T.sqr(sample_grad).sum())

        return rval


class AdversaryCost2(DefaultDataSpecsMixin, Cost):
    """
    """

    # Supplies own labels, don't get them from the dataset
    supervised = False

    def __init__(self, scale_grads=1, target_scale=.1,
            discriminator_default_input_include_prob = 1.,
            discriminator_input_include_probs=None,
            discriminator_default_input_scale=1.,
            discriminator_input_scales=None,
            generator_default_input_include_prob = 1.,
            generator_default_input_scale=1.,
            inference_default_input_include_prob=None,
            inference_input_include_probs=None,
            inference_default_input_scale=1.,
            inference_input_scales=None):
        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, **kwargs):
        S, d_obj, g_obj, i_obj = self.get_samples_and_objectives(model, data)
        return d_obj + g_obj + i_obj

    def get_samples_and_objectives(self, model, data):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        assert isinstance(model, AdversaryPair)
        g = model.generator
        d = model.discriminator

        # Note: this assumes data is design matrix
        X = data
        m = data.shape[space.get_batch_axis()]
        y1 = T.alloc(1, m, 1)
        y0 = T.alloc(0, m, 1)
        # NOTE: if this changes to optionally use dropout, change the inference
        # code below to use a non-dropped-out version.
        S, z = g.sample_and_noise(m, default_input_include_prob=self.generator_default_input_include_prob, default_input_scale=self.generator_default_input_scale)
        y_hat1 = d.dropout_fprop(X, self.discriminator_default_input_include_prob,
                                     self.discriminator_input_include_probs,
                                     self.discriminator_default_input_scale,
                                     self.discriminator_input_scales)
        y_hat0 = d.dropout_fprop(S, self.discriminator_default_input_include_prob,
                                     self.discriminator_input_include_probs,
                                     self.discriminator_default_input_scale,
                                     self.discriminator_input_scales)

        d_obj =  0.5 * (d.layers[-1].cost(y1, y_hat1) + d.layers[-1].cost(y0, y_hat0))
        g_obj = d.layers[-1].cost(y1, y_hat0)

        if model.inferer is not None:
            # Change this if we ever switch to using dropout in the
            # construction of S.
            S_nograd = block_gradient(S)  # Redundant as long as we have custom get_gradients
            z_hat = model.inferer.dropout_fprop(S_nograd, self.inference_default_input_include_prob,
                                                self.inference_input_include_probs,
                                                self.inference_default_input_scale,
                                                self.inference_input_scales)
            i_obj = model.inferer.layers[-1].cost(z, z_hat)
        else:
            i_obj = 0

        return S, d_obj, g_obj, i_obj

    def get_gradients(self, model, data, **kwargs):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        assert isinstance(model, AdversaryPair)
        g = model.generator
        d = model.discriminator

        S, d_obj, g_obj, i_obj = self.get_samples_and_objectives(model, data)

        g_params = g.get_params()
        d_params = d.get_params()
        for param in g_params:
            assert param not in d_params
        for param in d_params:
            assert param not in g_params
        d_grads = T.grad(d_obj, d_params)
        g_grads = T.grad(g_obj, g_params)

        if self.scale_grads:
            S_grad = T.grad(g_obj, S)
            scale = T.maximum(1., self.target_scale / T.sqrt(T.sqr(S_grad).sum()))
            g_grads = [g_grad * scale for g_grad in g_grads]

        rval = OrderedDict(safe_zip(d_params + g_params,
            d_grads + g_grads))
        if model.inferer is not None:
            i_params = model.inferer.get_params()
            i_grads = T.grad(i_obj, i_params)
            rval.update(OrderedDict(safe_zip(i_params, i_grads)))
        return rval, OrderedDict()

    def get_monitoring_channels(self, model, data, **kwargs):

        rval = OrderedDict()

        m = data.shape[0]

        g = model.generator
        d = model.discriminator

        y_hat = d.fprop(data)

        rval['false_negatives'] = T.cast((y_hat < 0.5).mean(), 'float32')

        samples = g.sample(m)
        y_hat = d.fprop(samples)
        rval['false_positives'] = T.cast((y_hat > 0.5).mean(), 'float32')
        # y = T.alloc(0., m, 1)
        cost = d.cost_from_X((samples, y_hat))
        sample_grad = T.grad(-cost, samples)
        rval['sample_grad_norm'] = T.sqrt(T.sqr(sample_grad).sum())
        _S, d_obj, g_obj, i_obj = self.get_samples_and_objectives(model, data)
        if i_obj != 0:
            rval['objective_i'] = i_obj
        rval['objective_d'] = d_obj
        rval['objective_g'] = g_obj
        return rval

def recapitate_discriminator(pair_path, new_head):
    pair = serial.load(pair_path)
    d = pair.discriminator
    del d.layers[-1]
    d.add_layers([new_head])
    return d

def theano_parzen(data, mu, sigma):
    """
    Credit: Yann N. Dauphin
    """
    x = data

    a = ( x.dimshuffle(0, 'x', 1) - mu.dimshuffle('x', 0, 1) ) / sigma

    E = log_mean_exp(-0.5*(a**2).sum(2))

    Z = mu.shape[1] * T.log(sigma * numpy.sqrt(numpy.pi * 2))

    #return theano.function([x], E - Z)
    return E - Z


def log_mean_exp(a):
    """
    Credit: Yann N. Dauphin
    """

    max_ = a.max(1)

    return max_ + T.log(T.exp(a - max_.dimshuffle(0, 'x')).mean(1))

class Sum(Layer):

    def __init__(self, layer_name):
        Model.__init__(self)
        self.__dict__.update(locals())
        del self.self
        self._params = []

    def set_input_space(self, space):
        self.input_space = space
        assert isinstance(space, CompositeSpace)
        self.output_space = space.components[0]

    def fprop(self, state_below):
        return sum(state_below)

    @functools.wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                    state=None, targets=None):
        rval = OrderedDict()

        if (state is not None) or (state_below is not None):
            if state is None:
                state = self.fprop(state_below)

            mx = state.max(axis=0)
            mean = state.mean(axis=0)
            mn = state.min(axis=0)
            rg = mx - mn

            rval['range_x_max_u'] = rg.max()
            rval['range_x_mean_u'] = rg.mean()
            rval['range_x_min_u'] = rg.min()

            rval['max_x_max_u'] = mx.max()
            rval['max_x_mean_u'] = mx.mean()
            rval['max_x_min_u'] = mx.min()

            rval['mean_x_max_u'] = mean.max()
            rval['mean_x_mean_u'] = mean.mean()
            rval['mean_x_min_u'] = mean.min()

            rval['min_x_max_u'] = mn.max()
            rval['min_x_mean_u'] = mn.mean()
            rval['min_x_min_u'] = mn.min()

        return rval
