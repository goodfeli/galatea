from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from theano.sandbox.rng_mrg import MRG_RandomStreams
from pylearn2.utils import sharedX
import numpy as np
import theano.tensor as T
from pylearn2.costs.cost import Cost
from theano.printing import Print
from pylearn2.space import CompositeSpace
from collections import OrderedDict
from pylearn2.models.mlp import MLP
from pylearn2.models.mlp import Layer
from pylearn2.models.mlp import Linear
from pylearn2.utils import safe_zip

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


class LinearAgents(Layer):
    def __init__(self, layer_name, **kwargs):
        self.layer_name = layer_name
        self.submodels = []

        def add_submodel(name):
            kwargs['layer_name'] = layer_name + '-' + name
            submodel = Linear(**kwargs)
            self.submodels.append(submodel)

        add_submodel('0')
        add_submodel('1')

    def set_mlp(self, mlp):
        self.mlp = mlp
        for submodel in self.submodels:
            submodel.set_mlp(mlp)

    def get_lr_scalers(self):
        rval = OrderedDict()

        for submodel in self.submodels:
            rval.update(submodel.get_lr_scalers())

        return rval

    def set_input_space(self, space):
        for submodel in self.submodels:
            submodel.set_input_space(space)
        self.input_space = space
        self.output_space = self.submodels[0].output_space

    def censor_updates(self, updates):
        for submodel in self.submodels:
            submodel.censor_updates(updates)

    def get_params(self):
        rval = []

        for model in self.submodels:
            rval += model.get_params()

        return rval

    def get_weights(self):
        W0 = self.submodels[0].get_weights()
        W1 = self.submodels[1].get_weights()
        rval = np.zeros((W0.shape[0], W0.shape[1] * 2))
        rval[:, 0::2] = W0
        rval[:, 1::2] = W1
        return rval

    def get_weights_view_shape(self):
        return (self.submodels[0].dim, 2)

    def get_weights_format(self):
        return ('v', 'h')

    def get_monitoring_channels(self):
        rval = OrderedDict()
        for i, submodel in enumerate(self.submodels):
            d = submodel.get_monitoring_channels()
            for key in d:
                rval[str(i) + '_' + key] = d[key]
        return rval

    def fprop(self, state_below):
        rval = self.submodels[1].fprop(state_below) > self.submodels[0].fprop(state_below)
        return rval


class AgentHive1(MLP):

    def __init__(self, **kwargs):
        MLP.__init__(self, **kwargs)
        for layer in self.layers[:-1]:
            assert type(layer) is LinearAgents # Yes, not isinstance

    def flip_fprop(self, state_below, return_all = False, flip_prob = 0.):

        rval = self.fprop(state_below, return_all)

        theano_rng = MRG_RandomStreams(2013 + 11 + 1)

        for i in xrange(len(rval) - 1):
            flip = theano_rng.binomial(p=flip_prob, size=rval[i].shape)
            rval[i] = (1-rval[i]) * flip + rval[i] * (1-flip)

        return rval

class AgentHive1Cost1(Cost):
    """
    """

    supervised = True

    def __init__(self, flip_prob):
        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, **kwargs):
        assert type(model) is AgentHive1
        space, sources = self.get_data_specs(model)
        space.validate(data)
        return model.cost_from_X(data)

    def get_gradients(self, model, data, **kwargs):
        assert type(model) is AgentHive1
        X, Y = data
        states = model.flip_fprop(X, return_all=True, flip_prob = self.flip_prob)
        classifier = model.layers[-1]
        cost_matrix = classifier.cost_matrix(Y_hat=states[-1], Y=Y)
        cost_vector = cost_matrix.sum(axis=1)
        hidden_states = states[:-1]
        last_hid = hidden_states[-1]

        classifier_params = classifier.get_params()
        classifier_grads = T.grad(cost_vector.mean(), classifier_params, consider_constant = [ last_hid ])

        rval = OrderedDict()
        for param, grad in zip(classifier_params, classifier_grads):
            rval[param] = grad

        reward_vector = - cost_vector

        mean_reward = sharedX(0.)

        new_mean_reward = reward_vector.mean()

        reward_vector -= mean_reward

        reward_vector = Print('reward_vector', attrs=['min', 'mean', 'max'])(reward_vector)

        for layer, ipt, opt in safe_zip(model.layers[:-1], [X] + hidden_states[:-1], hidden_states):
            ipt.name = 'ipt'
            opt.name = 'opt'
            target_matrix = reward_vector.dimshuffle(0, 'x') + T.alloc(0., reward_vector.shape[0], layer.submodels[0].dim)
            target_matrix.name = 'target_matrix'
            #opt = Print('opt', attrs=['min', 'max', 'mean'])(opt)
            for idx in xrange(2):
                if idx == 0:
                    mask = 1 - opt
                else:
                    mask = opt
                submodel = layer.submodels[idx]
                Y_hat = submodel.fprop(ipt)
                Y_hat.name = 'Y_hat_' + layer.layer_name + '_' + str(idx)
                Y_hat = Print(Y_hat.name, attrs=['min', 'mean', 'max'])(Y_hat)
                cost_matrix = submodel.cost_matrix(Y_hat = Y_hat, Y=target_matrix)
                from theano.printing import min_informative_str
                cost_matrix.name = 'orig_cost_matrix'
                cost_matrix = mask * cost_matrix
                # cost_matrix = Print('cost_matrix', attrs=['min', 'mean', 'max'])(cost_matrix)
                cost_matrix.name = 'masked_cost_matrix'
                params = submodel.get_params()
                weights = T.maximum(mask.sum(axis=0), 1)
                # weights = Print('weights', attrs=['min', 'mean', 'max'])(weights)
                weights.name = 'weights'
                cost_matrix = cost_matrix / weights # when we do the sum, we want it to be the mean across examples that affected the cost
                cost_matrix.name = 'weighted_cost_matrix'
                grads = T.grad(cost_matrix.sum(), params, consider_constant=[ipt, mask])
                for param, grad in zip(params, grads):
                    rval[param] = grad

        tc = .01

        return rval, OrderedDict([(mean_reward, tc * new_mean_reward + (1.-tc) * mean_reward)])


    def get_data_specs(self, model):
        space = CompositeSpace([model.get_input_space(), model.get_output_space()])
        sources = (model.get_input_source(), model.get_target_source())
        return (space, sources)
