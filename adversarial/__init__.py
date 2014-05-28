from theano.compat import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import tensor as T

from pylearn2.costs.cost import Cost
from pylearn2.costs.cost import DefaultDataSpecsMixin
from pylearn2.expr.nnet import kl
from pylearn2.models import Model
from pylearn2.utils import safe_zip

class AdversaryPair(Model):

    def __init__(self, generator, discriminator):
        self.__dict__.update(locals())
        del self.self

    def get_params(self):
        return self.generator.get_params() + self.discriminator.get_params()

    def get_input_space(self):
        return self.discriminator.get_input_space()

    def get_weights(self):
        return self.discriminator.get_weights()

    def get_weights_format(self):
        return self.discriminator.get_weights_format()

    def get_weights_view_shape(self):
        return self.discriminator.get_weights_view_shape()

    def get_monitoring_channels(self, data):
        rval = OrderedDict()

        assert data is None or data.ndim == 2

        g_ch = self.generator.get_monitoring_channels(data)
        # need to spoof targets: d_ch = self.discriminator.get_monitoring_channels(data)
        d_ch = OrderedDict()

        for key in g_ch:
            rval['gen_' + key] = g_ch[key]
        for key in d_ch:
            rval['dis_' + key] = d_ch[key]

        return rval


class Generator(Model):

    def __init__(self, mlp):
        self.__dict__.update(locals())
        del self.self
        self.theano_rng = MRG_RandomStreams(2014 * 5 + 27)

    def sample(self, num_samples, default_input_include_prob=1., default_input_scale=1.):
        # Assumes design matrix
        n = self.mlp.get_input_space().get_total_dimension()
        noise = self.theano_rng.normal(size=(num_samples, n), dtype='float32')
        return self.mlp.dropout_fprop(noise, default_input_include_prob=default_input_include_prob, default_input_scale=default_input_scale)

    def get_monitoring_channels(self, data):
        if data is None:
            m = 100
        else:
            m = data.shape[0]
        n = self.mlp.get_input_space().get_total_dimension()
        noise = self.theano_rng.normal(size=(m, n), dtype='float32')
        return self.mlp.get_monitoring_channels((noise, None))

    def get_params(self):
        return self.mlp.get_params()


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
        y = T.alloc(0., m, 1)
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
        discriminator_X = T.concatenate((X, S), axis=0)
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
        d_grads = T.grad(discrim_obj, d_params)

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
        y = T.alloc(0., m, 1)
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
        y1 = T.alloc(1, m, 1)
        y0 = T.alloc(0, m, 1)
        S = g.sample(m, default_input_include_prob=self.generator_default_input_include_prob, default_input_scale=self.generator_default_input_scale)
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
        y = T.alloc(0., m, 1)
        cost = d.cost_from_X((samples, y_hat))
        sample_grad = T.grad(-cost, samples)
        rval['sample_grad_norm'] = T.sqrt(T.sqr(sample_grad).sum())

        return rval
