import numpy as np

import theano
from theano import config
from theano import gof
from theano.printing import Print

from pylearn2.expr.nnet import arg_of_softmax
from pylearn2.models.model import Model
from pylearn2 import utils
from pylearn2.costs.cost import FixedVarDescr
from pylearn2.space import Conv2DSpace
from pylearn2.space import VectorSpace
from pylearn2.space import CompositeSpace
from pylearn2.utils import sharedX
from pylearn2.linear.conv2d import make_random_conv2D
from pylearn2.linear.conv2d import make_sparse_random_conv2D
from pylearn2.linear.conv2d_c01b import setup_detector_layer_c01b
from pylearn2.linear.matrixmul import MatrixMul
import theano.tensor as T
from pylearn2.expr.probabilistic_max_pooling import max_pool
from pylearn2.expr.probabilistic_max_pooling import max_pool_b01c
from pylearn2.expr.probabilistic_max_pooling import max_pool_c01b
from collections import OrderedDict
from pylearn2.utils import block_gradient
import warnings
from theano import function
from theano.sandbox.rng_mrg import MRG_RandomStreams
import time
from pylearn2.costs.cost import Cost
from pylearn2.utils import safe_zip
from pylearn2.utils import safe_izip
from pylearn2.utils import safe_union
io = None
from pylearn2.expr.probabilistic_max_pooling import max_pool_channels
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
from pylearn2.train_extensions import TrainExtension
from theano.gof.op import get_debug_values
from theano import printing


class SuperDBM(DBM):

    # Constructor is handled by DBM superclass


    def make_presynaptic_outputs(self, supervised):

        self.presynaptic_outputs = OrderedDict()
        self.visible_layer.install_presynaptic_outputs(self.presynaptic_outputs, self.batch_size)
        self.hidden_layers[-1].install_presynaptic_outputs(self.presynaptic_outputs, self.batch_size)

    def setup_inference_procedure(self):
        if not hasattr(self, 'inference_procedure') or \
                self.inference_procedure is None:
            self.inference_procedure = SuperWeightDoubling()
            self.inference_procedure.set_dbm(self)

    def add_polyak_channels(self, param_to_mean, monitoring_dataset):
        """
            Hack to make Polyak averaging work.
        """

        for param in param_to_mean:
            if type(param) != type(param_to_mean[param]):
                print type(param)
                print type(param_to_mean[param])
                assert False

        X = self.get_input_space().make_batch_theano()
        if isinstance(self.hidden_layers[-1], dbm.Softmax):
            Y = T.matrix()

            Y_hat = self.mf(X)[-1]

            for var in theano.gof.graph.ancestors([Y_hat]):
                if var.owner is not None:
                    node = var.owner
                    inputs = []
                    for inp in node.inputs:
                        if inp in param_to_mean:
                            inputs.append(param_to_mean[inp])
                        else:
                            inputs.append(inp)
                    node.inputs = inputs

            new_ancestors = theano.gof.graph.ancestors([Y_hat])

            for param in param_to_mean:
                assert param not in new_ancestors

            pred = T.argmax(Y_hat, axis=1)
            true = T.argmax(Y, axis=1)

            err = T.cast(T.neq(pred, true).mean(), X.dtype)

            assert isinstance(monitoring_dataset, dict)

            for dataset_name in monitoring_dataset:
                d = monitoring_dataset[dataset_name]

                if dataset_name == '':
                    channel_name = 'polyak_err'
                else:
                    channel_name = dataset_name + '_polyak_err'

                self.monitor.add_channel(name = channel_name,
                        val = err,
                        ipt = (X,Y),
                        dataset = d)


    def rao_blackwellize(self, layer_to_state):
        """
        layer_to_state:
            dict mapping layers to samples of that layer

        returns:
            layer_to_rao_blackwellized
                dict mapping layers to either samples or
                distributional samples.
                all hidden numbers with an even-numbered index
                become distributional.
        """

        layer_to_rao_blackwellized = OrderedDict()

        # Copy over visible layer
        layer_to_rao_blackwellized[self.visible_layer] = layer_to_state[self.visible_layer]

        for i, layer in enumerate(self.hidden_layers):
            if i % 2 == 0:
                # Even numbered layer, make distributional
                if i == 0:
                    layer_below = self.visible_layer
                else:
                    layer_below = self.hidden_layers[i-1]
                state_below = layer_to_state[layer_below]
                state_below = layer_below.upward_state(state_below)

                if i + 1 < len(self.hidden_layers):
                    layer_above = self.hidden_layers[i+1]
                    state_above = layer_to_state[layer_above]
                    state_above = layer_above.downward_state(state_above)
                else:
                    layer_above = None
                    state_above = None

                distributional = layer.mf_update(state_below = state_below,
                                                state_above = state_above,
                                                layer_above = layer_above,
                                                iter_name = 'rao_blackwell')
                layer_to_rao_blackwellized[layer] = distributional
            else:
                # Odd numbered layer, copy over
                layer_to_rao_blackwellized[layer] = layer_to_state[layer]

        assert all([layer in layer_to_rao_blackwellized for layer in layer_to_state])
        assert all([layer in layer_to_state for layer in layer_to_rao_blackwellized])

        return layer_to_rao_blackwellized


    def score_matching(self, X):
        """

        Returns the score matching objective for this model on a batch of
        examples X.

        Note:
        Score matching is also implemented in pylearn2.costs.ebm_estimation.
        However, that implementation is generic and based on scan. This
        method tries to look up an efficient model-specific implementation.
        Also, the version in ebm_estimation assumes X is a matrix, but here
        it is a theano batch in the visible layer's Space.

        Note:
        This method is used in conjunction with the
        pylearn2.costs.cost.make_method_cost function, so it should
        mimic the call signature of UnsupervisedCost.__call__
        (there is no "model" argument here because "model" is now "self").
        This means that it is important that the variable be named X and
        not V in order for named argument calls to MethodCost objects to work.
        """

        # eventually we should try to look up the right implementation
        # for now we just check if it is the only case I have implemented

        if isinstance(self.visible_layer, GaussianConvolutionalVisLayer) and \
            len(self.hidden_layers) == 1 and \
            isinstance(self.hidden_layers[0], ConvMaxPool):

            # see uper_dbm_score_matching.lyx

            warnings.warn("super_dbm score matching is untested."
                    "the math in the lyx file has not been verified."
                    "there is no test that the code matches the lyx file."
                    "Bad results on CIFAR-10 full image and CIFAR-10 patches "
                    "suggest there is a bug. See galatea/sparsity/cifar_grbm_super_sm.yaml for easiest case to debug.")
            assert False # really not a good idea to run this except to debug it

            vis = self.visible_layer
            hid, = self.hidden_layers

            V = X
            assert V.ndim == 4

            P, H = hid.mf_update(state_below = V,
                    state_above = None,
                    double_weights = False,
                    iter_name = 'rbm')

            assert H is hid.downward_state( (P,H) )

            recons = hid.downward_message(H) + vis.mu
            assert recons.ndim == 4

            beta = vis.beta

            # this should be non-negative
            hid_stuff = H * (1. - H)

            # this should be non-negative
            vis_stuff =  hid.transformer.lmul_sq_T(hid_stuff)

            sq_beta = T.sqr(beta)

            # this should be non-negative
            first_term_presum = sq_beta *(0.5* T.square(V-recons)+vis_stuff)
            first_term = first_term_presum.sum(axis=(1,2,3)).mean()
            assert first_term.ndim == 0

            second_term = - beta.sum()

            return first_term + second_term
        #end if gconv + convmaxpool
        raise NotImplementedError()
    #end score matching

from pylearn2.models.dbm import GaussianVisLayer




class DebugGaussianLayer(GaussianVisLayer):

    def get_params(self):
        if self.mu is None:
            return [self.cost_beta, self.beta]
        return [self.cost_beta, self.beta, self.mu]

    def broadcasted_cost_beta(self):
        """
        Returns beta, broadcasted to have the same shape as a batch of data
        """

        if self.tie_beta == 'locations':
            def f(x):
                if x == 'c':
                    return 0
                return 'x'
            axes = [f(ax) for ax in self.axes]
            rval = self.cost_beta.dimshuffle(*axes)
        else:
            assert self.tie_beta is None
            if self.nvis is None:
                axes = [0, 1, 2]
                axes.insert(self.axes.index('b'), 'x')
                rval = self.cost_beta.dimshuffle(*axes)
            else:
                rval = self.cost_beta.dimshuffle('x', 0)

        self.input_space.validate(rval)

        return rval

    def recons_cost(self, V, V_hat_unmasked, drop_mask = None, use_sum=False):

        V_hat = V_hat_unmasked

        assert V.ndim == V_hat.ndim
        beta = self.broadcasted_cost_beta()
        unmasked_cost = 0.5 * beta * T.sqr(V-V_hat) - 0.5*T.log(beta / (2*np.pi))
        assert unmasked_cost.ndim == V_hat.ndim

        if drop_mask is None:
            masked_cost = unmasked_cost
        else:
            masked_cost = drop_mask * unmasked_cost

        if use_sum:
            return masked_cost.mean(axis=0).sum()

        return masked_cost.mean()


# make old pickle files work
GaussianConvolutionalVisLayer = GaussianVisLayer

from pylearn2.models.dbm import ConvMaxPool
from pylearn2.models.dbm import ConvC01B_MaxPool

DenseMaxPool = BinaryVectorMaxPool



"""
class Verify(theano.gof.Op):
    hack used to make sure the right data is flowing through

    verify = { 0 : [0] }

    def make_node(self, xin):
        xout = xin.type.make_variable()
        return theano.gof.Apply(op=self, inputs=[xin], outputs=[xout])

    def __init__(self, X, name):
        self.X = X
        self.name = name

    def perform(self, node, inputs, output_storage):
        xin, = inputs
        xout, = output_storage
        xout[0] = xin

        X = self.X
        m = xin.shape[0]

        print self.name


        if not np.allclose(xin, X[0:m,:]):
            expected = X[0:m,:]
            if xin.shape != expected.shape:
                print 'xin.shape',xin.shape
                print 'expected.shape',expected.shape
            print 'max diff: ',np.abs(xin-expected).max()
            print 'min diff: ',np.abs(xin-expected).min()
            if xin.max() > X.max():
                print 'xin.max is too big for the dataset'
            if xin.min() < X.min():
                print 'xin.min() is too big for the dataset'
            if len(xin.shape) == 2:
                mean_xin = xin.mean(axis=1)
                mean_expected = expected.mean(axis=1)
                print 'Mean for each example:'
                print 'xin:'
                print mean_xin
                print 'expected:'
                print mean_expected
            if xin.shape[-1] == 3:
                from pylearn2.gui.patch_viewer import PatchViewer
                pv = PatchViewer((m,2),(X.shape[1],X.shape[2]),is_color=1)
                for i in xrange(m):
                    pv.add_patch(xin[i,:],rescale=True)
                    pv.add_patch(X[i,:],rescale=True)
                pv.show()
            print 'searching for xin in the dataset'
            for i in xrange(X.shape[0]):
                other = X[i,:]
                if np.allclose(xin[0,:], other):
                    print 'found alternate row match at idx ',i
        if self.name == 'features':
            assert False

    def c_code(self, *args, **kwargs):
        raise NotImplementedError()
"""

class Softmax(dbm.Softmax):

    def __init__(self, learn_init_inpainting_state=True, **kwargs):
        self.learn_init_inpainting_state=learn_init_inpainting_state
        super(Softmax, self).__init__(**kwargs)

    presynaptic_name = 'presynaptic_Y_hat'


    def ensemble_prediction(self, symbolic, outputs_dict, ensemble):

        Z = (outputs_dict[self.presynaptic_name] + symbolic) / ensemble.num_copies

        return T.nnet.softmax(Z)

    def ensemble_recons_cost(self, Y, Y_hat_unmasked, drop_mask_Y, scale, ensemble):
        return self.recons_cost(Y, Y_hat_unmasked, drop_mask_Y, scale)



def add_layers( super_dbm, new_layers ):
    """
        Modifies super_dbm to contain new_layers on top of its old hidden_layers
        Returns super_dbm

        This is mostly a convenience function for yaml files
    """

    super_dbm.add_layers( new_layers )

    return super_dbm


def unfreeze(augmented_dbm):
    augmented_dbm.freeze_lower = False
    return augmented_dbm


class AugmentedDBM(Model):
    """
        A DBM-like model.
        Contains one extra layer for transforming to the classification space.
        The 'mf' method of this class does not implement real mean field.
        Instead, it runs mean field in the DBM, then does one mean field update
        of the extra layer to get classification predictions.
        In other words, it defines a recurrent net that has feedback in the feature
        layers but passes through the classification layer only once.
    """

    def __init__(self, super_dbm, extra_layer, freeze_lower = False):
        """
            extra_layer:
                either an uninitialized layer to be added to the DBM, or
                the already initialized final layer of the DBM. In the latter
                case, we remove it from the DBM's inference feedback loop.
        """
        self.__dict__.update(locals())
        del self.self

        if extra_layer is super_dbm.hidden_layers[-1]:
            del super_dbm.hidden_layers[-1]
        else:
            extra_layer.set_dbm(super_dbm)
            extra_layer.set_input_space(super_dbm.hidden_layers[-1].get_output_space())
        self.force_batch_size = super_dbm.force_batch_size

        self.hidden_layers = [ extra_layer ]

    def get_weights_topo(self):
        return self.super_dbm.get_weights_topo()

    def get_weights(self):
        return self.super_dbm.get_weights()

    def get_weights_format(self):
        return self.super_dbm.get_weights_format()

    def get_weights_view_shape(self):
        return self.super_dbm.get_weights_view_shape()

    def get_params(self):
        if self.freeze_lower:
            return self.extra_layer.get_params()
        lower = self.super_dbm.get_params()
        for param in self.extra_layer.get_params():
            if param not in lower:
                lower.append(param)
        return lower

    def get_input_space(self):
        return self.super_dbm.get_input_space()

    def get_output_space(self):
        return self.extra_layer.get_output_space()

    def censor_updates(self, updates):
        self.super_dbm.censor_updates(updates)
        self.extra_layer.censor_updates(updates)

    def set_batch_size(self, batch_size):
        self.super_dbm.set_batch_size(batch_size)
        self.force_batch_size = self.super_dbm.force_batch_size
        self.extra_layer.set_batch_size(batch_size)

    def mf(self, V, return_history = False, **kwargs):

        #from pylearn2.config import yaml_parse
        #dataset = yaml_parse.load("""!obj:galatea.datasets.zca_dataset.ZCA_Dataset {
        #preprocessed_dataset: !pkl: "/data/lisa/data/cifar10/pylearn2_gcn_whitened/train.pkl",
        #preprocessor: !pkl: "/data/lisa/data/cifar10/pylearn2_gcn_whitened/preprocessor.pkl"
        #}""")
        #V = Verify(dataset.get_topological_view(),'data')(V)

        assert not return_history

        H_hat = self.super_dbm.mf(V, ** kwargs)[-1]
        upward_state = self.super_dbm.hidden_layers[-1].upward_state(H_hat)
        Y_hat = self.extra_layer.mf_update(state_below = upward_state)

        return [ Y_hat ]

def convert_to_augmented(super_dbm):
    return AugmentedDBM(super_dbm, super_dbm.hidden_layers[-1])


class SuperDBM_ConditionalNLL(Cost):

    supervised = True


    def __init__(self, grad_niter = None, block_grad = None):
        """
            grad_niter: Uses this many mean field iterations when computing the gradient.
                        When computing the cost value, we use model.niter
                        Extra steps cost more memory when computing the gradient but not
                        when computing the cost value.
        """
        self.__dict__.update(locals())

    def Y_hat(self, model, X, niter = None):
        assert isinstance(model.hidden_layers[-1], dbm.Softmax)
        Y_hat = model.mf(X, niter = niter, block_grad = self.block_grad)[-1]
        Y_hat.name = 'Y_hat'

        return Y_hat

    def __call__(self, model, X, Y, **kwargs):
        """ Returns - log P( Y | X) / m
            where Y is a matrix of one-hot labels,
            one label per row
            X is a batch of examples, X[i,:] being an example
            (but not necessarily a row, ie, could be an image)
            P is given by the model (see the __init__ docstring
            for details)
            m is the number of examples
        """

        if 'niter' in kwargs:
            niter = kwargs['niter']
        else:
            niter = None

        Y_hat = self.Y_hat(model, X, niter)
        assert Y_hat.ndim == 2
        assert Y.ndim == 2

        # Pull out the argument to the softmax
        arg = arg_of_softmax(Y_hat)
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

    def get_gradients(self, model, X, Y, record=None,
            ** kwargs):

        new_kwargs = { 'niter' : self.grad_niter }
        new_kwargs.update(kwargs)

        cost = self(model, X, Y, ** new_kwargs)

        if record is not None:
            record.handle_line("SuperDBM_NLL.get_gradients cost " \
                + printing.var_descriptor(cost)+'\n')

        params = model.get_params()
        assert isinstance(params, list)
        grads = T.grad(cost, params, disconnected_inputs='ignore')

        if record is not None:
            for param, grad in zip(params, grads):
                record.handle_line("SuperDBM_NLL.get_gradients param order check "\
                        + printing.var_descriptor(param)+'\n')
                record.handle_line("SuperDBM_NLL.get_gradients grad check "\
                        + printing.var_descriptor(grad)+'\n')

        grads = OrderedDict(safe_zip(params, grads))

        return grads, OrderedDict()

    def get_monitoring_channels(self, model, X, Y, **kwargs):

        Y_hat = self.Y_hat(model, X)

        Y = T.argmax(Y, axis=1)
        Y = T.cast(Y, Y_hat.dtype)

        argmax = T.argmax(Y_hat,axis=1)
        if argmax.dtype != Y_hat.dtype:
            argmax = T.cast(argmax, Y_hat.dtype)
        neq = T.neq(Y , argmax).mean()
        if neq.dtype != Y_hat.dtype:
            neq = T.cast(neq, Y_hat.dtype)
        acc = 1.- neq

        assert acc.dtype == Y_hat.dtype

        return { 'acc' : acc, 'err' : 1. - acc }

def ditch_mu(model):
    model.visible_layer.mu = None
    return model

class _DummyVisible(Layer):
    """ Hack used by LayerAsClassifier"""
    def upward_state(self, total_state):
        return total_state

    def get_params(self):
        return set([])
class LayerAsClassifier(SuperDBM):
    """ A hack that lets us use a SuperDBM layer
    as a single-layer classifier without needing
    to set up an explicit visible layer object, etc."""

    def __init__(self, layer, nvis):
        self.__dict__.update(locals())
        del self.self
        self.space = VectorSpace(nvis)
        layer.set_input_space(self.space)
        self.hidden_layers = [ layer ]


        self.visible_layer = _DummyVisible()

    def get_input_space(self):
        return self.space

from pylearn2.models.dbm import CompositeLayer


BinaryVector = dbm.BinaryVector




class MF_L1_ActCost(Cost):
    """
    L1 activation cost on the mean field parameters.

    Adds a cost of:

    coeff * max( abs(mean_activation - target) - eps, 0)

    averaged over units

    for each layer.

    """

    def __init__(self, targets, coeffs, eps, supervised):
        """
        targets: a list, one element per layer, specifying the activation
                each layer should be encouraged to have
                    each element may also be a list depending on the
                    structure of the layer.
                See each layer's get_l1_act_cost for a specification of
                    what the state should be.
        coeffs: a list, one element per layer, specifying the coefficient
                to put on the L1 activation cost for each layer
        supervised: If true, runs mean field on both X and Y, penalizing
                the layers in between only
        """
        self.__dict__.update(locals())
        del self.self

    def __call__(self, model, X, Y = None, ** kwargs):

        if self.supervised:
            assert Y is not None
            H_hat = model.mf(X, Y= Y)
        else:
            H_hat = model.mf(X)

        hidden_layers = model.hidden_layers
        if self.supervised:
            hidden_layers = hidden_layers[:-1]
            H_hat = H_hat[:-1]

        layer_costs = []
        for layer, mf_state, targets, coeffs, eps in \
            safe_zip(hidden_layers, H_hat, self.targets, self.coeffs, self.eps):
            cost = None
            try:
                cost = layer.get_l1_act_cost(mf_state, targets, coeffs, eps)
            except NotImplementedError:
                assert isinstance(coeffs, float) and coeffs == 0.
                assert cost is None # if this gets triggered, there might
                    # have been a bug, where costs from lower layers got
                    # applied to higher layers that don't implement the cost
                cost = None
            if cost is not None:
                layer_costs.append(cost)


        assert T.scalar() != 0. # make sure theano semantics do what I want
        layer_costs = [ cost for cost in layer_costs if cost != 0.]

        if len(layer_costs) == 0:
            return T.as_tensor_variable(0.)
        else:
            total_cost = reduce(lambda x, y: x + y, layer_costs)
        total_cost.name = 'MF_L1_ActCost'

        assert total_cost.ndim == 0

        return total_cost


from pylearn2.costs.dbm import WeightDecay as DBM_WeightDecay

class StochasticWeightDecay(Cost):

    def __init__(self, coeffs, include_prob):
        self.__dict__.update(locals())
        del self.self

    def __call__(self, model, X, Y=None, swd_masks=None, **kwargs):
        weights = self.get_weights(model)

        total_cost = 0.

        for c, w, m in safe_zip(self.coeffs, weights, swd_masks):
            total_cost += c * T.sqr(w*m).sum()

        total_cost.name = 'stochastic_weight_decay'

        return total_cost

    def get_weights(self, model):

        rval = []

        for layer in model.hidden_layers:
            params = layer.get_params()
            weights = [ param for param in params if param.ndim == 2]
            assert len(weights) == 1
            rval.append(weights[0])

        return rval

    def get_fixed_var_descr(self, model, X, Y):

        rval = FixedVarDescr()

        masks = [sharedX(w.get_value() * 0.) for w in self.get_weights(model)]

        rval.fixed_vars = {'swd_masks' : masks}

        theano_rng = MRG_RandomStreams(201255319)

        updates = [(m, theano_rng.binomial(p=self.include_prob, size=m.shape, dtype=config.floatX))
                for m in masks]

        rval.on_load_batch = [ utils.function([X, Y], updates=updates) ]

        return rval





def set_niter(super_dbm, niter):
    super_dbm.niter = niter
    return super_dbm

def load_matlab_dbm(path):
    """ Loads a two layer DBM stored in the format used by Ruslan Salakhutdinov's
    matlab demo.

    This function can only load the model after it has been put together for fine
    tuning. Loading the model at the end of layerwise pretraining doesn't make sense
    because the first layer RBM's hidden units and the second layer RBM's visible units
    specify different biases for the same variable in the DBM.
    """

    # Lazy import because scipy causes trouble on travisbot
    global io
    if io is None:
        from scipy import io

    d = io.loadmat(path)

    # Format the data
    for key in d:
        try:
            d[key] = np.cast[config.floatX](d[key])
        except:
            pass

    # Visible layer
    visbiases = d['visbiases']
    assert len(visbiases.shape) == 2
    assert visbiases.shape[0] == 1
    visbiases = visbiases[0,:]

    vis_layer = BinaryVisLayer(nvis = visbiases.shape[0])
    vis_layer.set_biases(visbiases)

    # Set up hidden layers
    hidden_layers = []
    Ws = []
    bs = []

    bias0 = 'hidbiases'
    bias1 = 'penbiases'
    weights0 = 'vishid'
    weights1 = 'hidpen'
    weightsc = 'labpen'

    # First hidden layer
    hidbiases = d[bias0]
    assert len(hidbiases.shape) == 2
    assert hidbiases.shape[0] == 1
    hidbiases = hidbiases[0,:]

    vishid = d[weights0]

    hid0 = DenseMaxPool(detector_layer_dim = hidbiases.shape[0],
            pool_size = 1, irange = 0., layer_name = 'h0')

    hidden_layers.append(hid0)
    Ws.append(vishid)
    bs.append(hidbiases)

    # Second hidden layer
    penbiases = d['penbiases']
    assert len(penbiases.shape) == 2
    assert penbiases.shape[0] == 1
    penbiases = penbiases[0,:]
    hidpen = d['hidpen']

    hid1 = DenseMaxPool(detector_layer_dim = penbiases.shape[0],
        pool_size = 1, irange = 0., layer_name = 'h1')

    hidden_layers.append(hid1)
    Ws.append(hidpen)
    bs.append(penbiases)

    # Class layer
    labbiases = d['labbiases']
    assert len(labbiases.shape) == 2
    assert labbiases.shape[0] == 1
    labbiases = labbiases[0,:]

    if len(hidden_layers) == 1:
        W = d['labhid'].T
    else:
        W = d['labpen'].T

    c = Softmax(n_classes = labbiases.shape[0],
        irange = 0., layer_name = 'c')

    hidden_layers.append(c)
    Ws.append(W)
    bs.append(labbiases)

    #hidden_layers = hidden_layers[0:1]
    #Ws = Ws[0:1]
    #bs = bs[0:1]

    dbm = SuperDBM(batch_size = 100,
                    visible_layer = vis_layer,
                    hidden_layers = hidden_layers,
                    niter = 50)

    for W, b, layer in safe_zip(Ws, bs, dbm.hidden_layers):
        layer.set_weights(W)
        layer.set_biases(b)

    dbm.dataset_yaml_src = """
    !obj:pylearn2.datasets.mnist.MNIST {
        which_set : 'train',
        binarize  : 1,
        one_hot   : 1
    }"""

    return dbm

def zero_last_weights(super_dbm, niter):
    super_dbm.niter = niter
    super_dbm.hidden_layers[-1].set_weights(super_dbm.hidden_layers[-1].get_weights() * 0)
    return super_dbm

class MLP_Wrapper(Model):

    def __init__(self, super_dbm, decapitate = True, final_irange = None,
            initially_freeze_lower = False, decapitated_value = None,
            train_rnn_y = False, gibbs_features = False, top_down = False,
            copy_constraints = False, modify_input=0):

        # Note: this doesn't handle the 'copies' feature very well.
        # The best way to fit it is probably to write the mf method more generically

        assert initially_freeze_lower in [True, False, 0, 1]
        assert decapitate in [True, False, 0, 1]
        assert train_rnn_y in [True, False, 0, 1]
        self.__dict__.update(locals())

        model = super_dbm
        if hasattr(model.visible_layer, 'center') and model.visible_layer.center:
            self.v_ofs = model.visible_layer.offset
        else:
            self.v_ofs = 0

        if model.hidden_layers[0].center:
            self.h1_ofs = model.hidden_layers[0].offset
        else:
            self.h1_ofs = 0

        if model.hidden_layers[1].center:
            self.h2_ofs = model.hidden_layers[1].offset
        else:
            self.h2_ofs = 0

        if model.hidden_layers[2].center:
            self.y_ofs = model.hidden_layers[2].offset
        else:
            self.y_ofs = 0


        if decapitate:
            if decapitated_value is None:
                decapitated_value = 0.
        else:
            assert decapitated_value is None

        if gibbs_features:
            assert not train_rnn_y

        self.force_batch_size = super_dbm.force_batch_size
        if len(super_dbm.hidden_layers) == 3:
            self.orig_sup = True
            l1, l2, c = super_dbm.hidden_layers
        else:
            self.orig_sup = False
            assert not decapitate # can't decapitate the already headless
            assert final_irange is not None
            l1, l2 = super_dbm.hidden_layers
            c = None
        assert isinstance(l1, DenseMaxPool)
        assert isinstance(l2, DenseMaxPool)
        if self.orig_sup:
            assert isinstance(c, dbm.Softmax)

        self._params = []

        # Layer 1
        W = l1.get_weights()
        vis = super_dbm.visible_layer
        if hasattr(vis, 'beta'):
            beta = vis.beta.get_value()
            W = (beta * W.T).T
        self.vishid = sharedX(W, 'vishid')
        self._params.append(self.vishid)
        self.hidbias = sharedX(l1.get_biases(), 'hidbias')
        self._params.append(self.hidbias)

        # Layer 2
        if not hasattr(l1, 'copies'):
            l1.copies = 1
        if not hasattr(l2, 'copies'):
            l2.copies = 1
        self.hidpen = sharedX(l2.get_weights()*l1.copies, 'hidpen')
        self._params.append(self.hidpen)
        self.penhid = sharedX(l2.get_weights().T*l2.copies, 'penhid')
        self._params.append(self.penhid)
        penbias = l2.get_biases()
        if decapitate:
            Wc = c.get_weights() * l2.copies
            penbias += np.dot(Wc,
                    np.ones((c.n_classes,), dtype = penbias.dtype) * decapitated_value / c.n_classes)
            l2.set_biases(penbias)
        self.penbias = sharedX(l2.get_biases(), 'penbias')
        self._params.append(self.penbias)

        # Class layer
        if decapitate:
            self.c = c
            del super_dbm.hidden_layers[-1]
        else:
            if c is not None and hasattr(c, 'copies'):
                c_copies = c.copies
            else:
                c_copies = 1
            self.c = Softmax(n_classes = 10, irange = 0., layer_name = 'final_output', copies = c_copies,
                    center = model.hidden_layers[-1].center)
            self.c.dbm = l1.dbm
            self.c.set_input_space(l2.get_output_space())
            if self.orig_sup:
                self.c.set_weights(c.get_weights() * l2.copies)
                self.c.set_biases(c.get_biases())
        self._params.extend(self.c.get_params())

        if train_rnn_y:
            assert not decapitate
            self._params = safe_union(self._params, c.get_params())

        if final_irange is not None:
            W = self.c.dbm.rng.uniform(-final_irange, final_irange,
                    (self.c.input_space.get_total_dimension(),
                        self.c.n_classes)) * l2.copies
            self.c.set_weights(W.astype('float32'))
            self.c.set_biases(np.zeros((self.c.n_classes)).astype('float32'))

        self.hidden_layers = [ self.c]

        if top_down:
            assert not decapitate
            assert self.orig_sup
            self.labpen = sharedX(self.c.get_weights().T, 'labpen')
            self._params.append(self.labpen)

        self.max_col_norms = {}
        self.max_row_norms = {}
        if copy_constraints:
            # note: this really does not play nice with the "copies" feature
            self.max_col_norms[self.vishid] = l1.max_col_norm
            self.max_col_norms[self.hidpen] = l2.max_col_norm
            self.max_row_norms[self.penhid] = l2.max_col_norm
            self.max_col_norms[self.c.W] = c.max_col_norm
            if hasattr(self, 'labpen'):
                self.max_row_norms[self.labpen] = c.max_col_norm


        if initially_freeze_lower:
            lr_scalers = OrderedDict()
            gate = sharedX(0.)
            for param in self._params:
                if param not in self.c.get_params():
                    lr_scalers[param] = gate
            self.lr_scalers = lr_scalers
        else:
            self.lr_scalers = OrderedDict()


    """
    def get_monitoring_channels(self, X, Y = None, ** kwargs):

        V = X

        q = self.super_dbm.mf(V, ** kwargs)

        Y_hat = q[-1]

        y = T.argmax(Y, axis=1)
        y_hat = T.argmax(Y_hat, axis=1)

        misclass = T.neq(y, y_hat).mean()

        rval = OrderedDict()

        rval['raw_dbm_misclass'] = T.cast(misclass, 'float32')

        return rval
    """

    def censor_updates(self, updates):

        for W in self.max_col_norms:
            max_col_norm = self.max_col_norms[W]
            if W in updates:
                updated_W = updates[W]
                col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                desired_norms = T.clip(col_norms, 0, max_col_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))
        for W in self.max_row_norms:
            max_row_norm = self.max_row_norms[W]
            if W in updates:
                updated_W = updates[W]
                row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=1))
                desired_norms = T.clip(row_norms, 0, max_row_norm)
                updates[W] = updated_W * ((desired_norms / (1e-7 + row_norms)).dimshuffle(0, 'x'))



    def get_lr_scalers(self):
        return self.lr_scalers

    def mf(self, V, return_history = False, ** kwargs):
        assert not return_history
        if not hasattr(self, 'gibbs_features'):
            self.gibbs_features = False
        if not self.gibbs_features:
            if self.modify_input:
                V, q = self.super_dbm.inference_procedure.multi_infer(V, return_history=True, **kwargs)[-1]
            else:
                q = self.super_dbm.mf(V, ** kwargs)
        else:
            theano_rng = MRG_RandomStreams(42)
            layer_to_state = { self.super_dbm.visible_layer : V }
            qs = []
            m = 10
            for i in xrange(m):
                for layer in self.super_dbm.hidden_layers:
                    layer_to_state[layer] = layer.get_total_state_space().get_origin_batch(self.super_dbm.batch_size)
                    if isinstance(layer_to_state[layer], tuple):
                        layer_to_state[layer] = tuple([T.as_tensor_variable(elem.astype('float32')) for elem in layer_to_state[layer]])
                    else:
                        layer_to_state[layer] = T.as_tensor_variable(elem.astype('float32'))
                layer_to_updated = self.super_dbm.mcmc_steps(layer_to_state, theano_rng, layer_to_clamp = { self.super_dbm.visible_layer: 1 },
                    num_steps = 6)
                q = [ layer_to_updated[layer] for layer in self.super_dbm.hidden_layers]
                for j in xrange(len(q)):
                    if isinstance(q[j], tuple):
                        q[j] = q[j][0]
                qs.append(q)
            q = [ sum([subq[i] for subq in qs])/m for i in xrange(len(qs[0])) ]
            q[0] = (q[0], q[0])
            q[1] = (q[1], q[1])
        if not hasattr(self, 'decapitate'):
            self.decapitate = True

        if self.decapitate or not self.orig_sup:
            _, H2 = q
        else:
            _, H2, y = q
            assert y.ndim == 2

        _, H2 = H2


        below = T.dot((V - self.v_ofs), self.vishid)
        above = T.dot((H2 - self.h2_ofs), self.penhid)
        H1 = T.nnet.sigmoid(below + above + self.hidbias)
        if self.top_down:
            top_down = T.dot((y - self.y_ofs), self.labpen) * self.c.copies
        else:
            top_down = 0.
        H2 = T.nnet.sigmoid(T.dot((H1 - self.h1_ofs), self.hidpen) + top_down + self.penbias)
        Y = self.c.mf_update(state_below = H2)

        return [ Y ]


    def mf_missing(self, V, missing_mask, return_history = False, ** kwargs):
        assert not return_history
        if not hasattr(self, 'gibbs_features'):
            self.gibbs_features = False
        if not hasattr(self, 'modify_input'):
            self.modify_input = False
        if not self.gibbs_features:
            if self.modify_input:
                raise NotImplementedError()
            else:
                history = self.super_dbm.inference_procedure.do_inpainting(V=V, drop_mask=missing_mask, return_history=True)
                final_state = history[-1]
                V = final_state['V_hat']
                q = final_state['H_hat']
        else:
            raise NotImplementedError()

        if not hasattr(self, 'decapitate'):
            self.decapitate = True

        if not hasattr(self, 'v_ofs'):
            self.v_ofs = 0.
            self.h1_ofs = 0.
            self.h2_ofs = 0.
            self.y_ofs = 0.

        if self.decapitate or not self.orig_sup:
            _, H2 = q
        else:
            _, H2, y = q
            assert y.ndim == 2

        _, H2 = H2


        below = T.dot((V - self.v_ofs), self.vishid)
        above = T.dot((H2 - self.h2_ofs), self.penhid)
        H1 = T.nnet.sigmoid(below + above + self.hidbias)
        if self.top_down:
            top_down = T.dot((y - self.y_ofs), self.labpen) * self.c.copies
        else:
            top_down = 0.
        H2 = T.nnet.sigmoid(T.dot((H1 - self.h1_ofs), self.hidpen) + top_down + self.penbias)
        Y = self.c.mf_update(state_below = H2)

        return [ Y ]

    def set_batch_size(self, batch_size):
        self.super_dbm.set_batch_size(batch_size)
        self.c.set_batch_size(batch_size)
        self.force_batch_size = self.super_dbm.force_batch_size

    def get_input_space(self):
        return self.super_dbm.get_input_space()

    def get_output_space(self):
        return self.super_dbm.get_output_space()

    def get_weights(self):
        print 'MLP weights'
        return self.vishid.get_value()

    def get_weights_format(self):
        return ('v','h')

class DeepMLP_Wrapper(Model):

    def __init__(self, super_dbm, decapitate = True,
            decapitated_value = None,
            inference_type = 0
            ):


        self.stripped_down = False

        assert decapitate in [True, False, 0, 1]
        self.__dict__.update(locals())

        if decapitate:
            if decapitated_value is None:
                decapitated_value = 0.
        else:
            assert decapitated_value is None


        self.force_batch_size = super_dbm.force_batch_size
        l1, l2, l3, c = super_dbm.hidden_layers

        assert isinstance(l1, DenseMaxPool)
        assert isinstance(l2, DenseMaxPool)
        assert isinstance(l3, DenseMaxPool)
        assert isinstance(c, dbm.Softmax)

        self._params = []

        # Layer 1
        self.vis_h0 = sharedX(l1.get_weights(), 'vis_h0')
        self._params.append(self.vis_h0)
        self.h0_bias = sharedX(l1.get_biases(), 'h0_bias')
        self._params.append(self.h0_bias)
        if hasattr(l1, 'mask'):
            assert False # debugging, remove if surprising
            self.vis_h0_mask = l1.mask
        else:
            self.vis_h0_mask = None

        # Layer 2
        self.h0_h1 = sharedX(l2.get_weights(), 'h0_h1')
        self._params.append(self.h0_h1)
        self.h1_h0 = sharedX(l2.get_weights().T, 'h1_h0')
        self._params.append(self.h1_h0)
        self.h1_bias = sharedX(l2.get_biases(), 'h1_bias')
        self._params.append(self.h1_bias)
        if hasattr(l2, 'mask'):
            self.h0_h1_mask = l2.mask
        else:
            self.h0_h1_mask = None

        # Layer 3
        self.h1_h2 = sharedX(l3.get_weights(), 'h1_h2')
        self._params.append(self.h1_h2)
        self.h2_h1 = sharedX(l3.get_weights().T, 'h2_h1')
        self._params.append(self.h2_h1)
        penbias = l3.get_biases()
        if decapitate:
            Wc = c.get_weights()
            penbias += np.dot(Wc,
                    np.ones((c.n_classes,), dtype = penbias.dtype) * decapitated_value / c.n_classes)
            l3.set_biases(penbias)
        self.penbias = sharedX(l3.get_biases(), 'h2_bias')
        self._params.append(self.penbias)
        if hasattr(l3, 'mask'):
            self.h1_h2_mask = l3.mask
        else:
            self.h1_h2_mask = None

        # Class layer
        if decapitate:
            self.c = c
            del super_dbm.hidden_layers[-1]
        else:
            self.c = Softmax(n_classes = 10, irange = 0., layer_name = 'final_output')
            self.c.dbm = l1.dbm
            self.c.set_input_space(l3.get_output_space())
            self.c.set_weights(c.get_weights())
            self.c.set_biases(c.get_biases())
        if inference_type == 1:
            self.y_h2 = sharedX(self.c.get_weights().T)
            self._params.append(self.y_h2)
        self._params = safe_union(self._params, self.c.get_params())
        self.hidden_layers = [ self.c ]

    def censor_updates(self, updates):

        def apply_mask(param, mask):
            if mask is not None and param in updates:
                updates[param] = updates[param] * mask

        def transpose(x):
            if x is None:
                return x
            return x.T

        params = [ self.vis_h0, self.h0_h1, self.h1_h0, self.h1_h2 ]
        masks  = [ self.vis_h0_mask, self.h0_h1_mask, transpose(self.h0_h1_mask), self.h1_h2_mask]

        for param, mask in safe_zip(params, masks):
            apply_mask(param, mask)

    def strip_down(self):
        self.stripped_down = True
        del self.super_dbm

    def dump_func(self):

        V = T.matrix()
        q = self.super_dbm.mf(V)

        mod_V = self.super_dbm.visible_layer.upward_state(V)

        if self.decapitate:
            _, H1, H2 = q
            _, H1 = H1
            _, H2 = H2
            outputs = [mod_V, H1, H2]
        else:
            _, H1, H2, y = q
            _, H1 = H1
            _, H2 = H2
            outputs = [mod_V, H1, H2, y]

        output = T.concatenate(tuple(outputs), axis=1)

        return function([V], output)


    def mf(self, V, return_history = False, ** kwargs):
        assert not return_history

        if not self.stripped_down:
            q = self.super_dbm.mf(V, ** kwargs)
            V = self.super_dbm.visible_layer.upward_state(V)
            if self.decapitate:
                _, H1, H2 = q
            else:
                _, H1, H2, y = q
            _, H1 = H1
            _, H2 = H2
        else:
            packed = V
            num_V = self.vis_h0.get_value().shape[0]
            V = packed[:,:num_V]
            pos = num_V
            num_H1 = self.h1_h2.get_value().shape[0]
            H1 = packed[:,pos:pos+num_H1]
            pos += num_H1
            num_H2 = self.h1_h2.get_value().shape[1]
            H2 = packed[:,pos:pos+num_H2]
            pos += num_H2
            if not self.decapitate:
                y = packed[:,pos:]


        if self.inference_type == 0:
            below = T.dot(V, self.vis_h0)
            above = T.dot(H1, self.h1_h0)
            H0 = T.nnet.sigmoid(below + above + self.h0_bias)
            below = T.dot(H0, self.h0_h1)
            above = T.dot(H2, self.h2_h1)
            H1 = T.nnet.sigmoid(below + above + self.h1_bias)
            below = T.dot(H1, self.h1_h2)
            H2 = T.nnet.sigmoid(below + self.penbias)
            Y = self.c.mf_update(state_below = H2)
        elif self.inference_type == 1:
            below = T.dot(V, self.vis_h0)
            above = T.dot(H1, self.h1_h0)
            H0 = T.nnet.sigmoid(below + above + self.h0_bias)
            below = T.dot(H0, self.h0_h1)
            above = T.dot(H2, self.h2_h1)
            H1 = T.nnet.sigmoid(below + above + self.h1_bias)
            below = T.dot(H1, self.h1_h2)
            above = T.dot(y, self.y_h2)
            H2 = T.nnet.sigmoid(below + above + self.penbias)
            Y = self.c.mf_update(state_below = H2)
        else:
            raise ValueError()

        return [ Y ]

    def set_batch_size(self, batch_size):
        if not self.stripped_down:
            self.super_dbm.set_batch_size(batch_size)
            self.force_batch_size = self.super_dbm.force_batch_size
        self.c.set_batch_size(batch_size)

    def get_input_space(self):
        if self.stripped_down:
            return VectorSpace(self.vis_h0.get_value().shape[0])
        return self.super_dbm.get_input_space()

    def get_output_space(self):
        return self.c.get_output_space()

    def get_weights(self):
        print 'MLP weights'
        return self.vis_h0.get_value()

    def get_weights_format(self):
        return ('v','h')

class MatrixDecay(Cost):
    def __init__(self, coeff):
        self.coeff = coeff

    def __call__(self, model, X, Y = None, ** kwargs):
        return self.coeff * sum([ T.sqr(param).sum() for param in model.get_params() if param.ndim == 2])

class ActivateLower(TrainExtension):
    def on_monitor(self, model, dataset, algorithm):
        if model.monitor.get_epochs_seen() == 6:
            lr_scalers = model.lr_scalers
            values = lr_scalers.values()
            assert all([value is values[0] for value in values])
            values[0].set_value(np.cast[config.floatX](1.))


class UnrollUntie(Model):

    def __init__(self, super_dbm, niter):
        self.__dict__.update(locals())
        del self.self
        self.input_space = super_dbm.get_input_space()
        self.output_space = super_dbm.get_output_space()

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
        for name in param_names:
            sh = [ sharedX(locals()[name]) for i in xrange(niter) ]
            setattr(self, name, sh)
        self.penhid[0] = None
        self.labpen[0] = None
        self._params = []
        for name in param_names:
            self._params.extend([elem for elem in getattr(self, name) if elem is not None])
        self.hidden_layers = super_dbm.hidden_layers


    def set_batch_size(self, batch_size):
        self.force_batch_size = batch_size

    def mf(self, V, return_history = False, niter = None, block_grad = None):
        assert return_history is False
        assert niter is None
        assert block_grad is None

        H1 = T.nnet.sigmoid(T.dot(V, 2. * self.vishid[0]) + self.biashid[0])
        H2 = T.nnet.sigmoid(T.dot(H1, 2 * self.hidpen[0]) + self.biaspen[0])
        Y = T.nnet.softmax(T.dot(H2, self.penlab[0]) + self.biaslab[0])

        for i in xrange(1, self.niter):
            H1 = T.nnet.sigmoid(T.dot(V, self.vishid[i])+T.dot(H2, self.penhid[i])+self.biashid[i])
            Y = T.nnet.softmax(T.dot(H2, self.penlab[i]) + self.biaslab[i])
            H2 = T.nnet.sigmoid(T.dot(H1, self.hidpen[i]) + T.dot(Y, self.labpen[i]) + self.biaspen[0])

        return [H1, H2, Y]




class Dropout(InferenceProcedure):
    """
    An InferenceProcedure that initializes the mean field parameters based on the
    biases in the model. This InferenceProcedure uses the same weights at every
    iteration, rather than doubling the weights on the first pass.
    """

    def set_dbm(self, dbm):
        self.dbm = dbm
        def dropout_mask_like(x):
            return sharedX(0. * x.get_value())

        def dropout_structure(x):
            if isinstance(x, (list, tuple)):
                return [ dropout_structure(elem) for elem in x]
            return dropout_mask_like(x)

        if not hasattr(self, 'V_dropout'):
            self.V_dropout = dropout_structure(dbm.visible_layer.make_state(dbm.batch_size, dbm.rng))
            H_hat_states = [layer.make_state(dbm.batch_size, dbm.rng) for layer in dbm.hidden_layers]
            self.H_dropout = dropout_structure(H_hat_states)

    def __init__(self, include_prob=0.5, include_prob_V = 1., include_prob_Y = 1.):
        self.__dict__.update(locals())
        del self.self

    def mf(self, V, Y = None, return_history = False, niter = None, block_grad = None):

        dbm = self.dbm

        assert Y not in [True, False, 0, 1]
        assert return_history in [True, False, 0, 1]

        if Y is not None:
            dbm.hidden_layers[-1].get_output_space().validate(Y)

        if niter is None:
            niter = dbm.niter

        H_hat = [None] + [layer.init_mf_state() for layer in dbm.hidden_layers[1:]]

        # Make corrections for if we're also running inference on Y
        if Y is not None:
            # Last layer is clamped to Y
            H_hat[-1] = Y

        history = [ list(H_hat) ]

        #we only need recurrent inference if there are multiple layers
        assert (niter > 1) == (len(dbm.hidden_layers) > 1)

        for i in xrange(niter):
            for j in xrange(0,len(H_hat),2):
                if j == 0:
                    state_below = dbm.visible_layer.upward_state(V)
                else:
                    state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                if j == len(H_hat) - 1:
                    state_above = None
                    layer_above = None
                else:
                    state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                    layer_above = dbm.hidden_layers[j+1]
                H_hat[j] = dbm.hidden_layers[j].mf_update(
                        state_below = state_below,
                        state_above = state_above,
                        layer_above = layer_above)

            if Y is not None:
                H_hat[-1] = Y

            for j in xrange(1,len(H_hat),2):
                state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                if j == len(H_hat) - 1:
                    state_above = None
                    state_above = None
                else:
                    state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                    layer_above = dbm.hidden_layers[j+1]
                H_hat[j] = dbm.hidden_layers[j].mf_update(
                        state_below = state_below,
                        state_above = state_above,
                        layer_above = layer_above)
                #end ifelse
            #end for odd layer

            if Y is not None:
                H_hat[-1] = Y

            for i, elem in enumerate(H_hat):
                if elem is Y:
                    assert i == len(H_hat) -1
                    continue
                else:
                    assert elem not in history[-1]


            if block_grad == i + 1:
                H_hat = block(H_hat)

            history.append(list(H_hat))
        # end for mf iter

        # Run some checks on the output
        for layer, state in safe_izip(dbm.hidden_layers, H_hat):
            upward_state = layer.upward_state(state)
            layer.get_output_space().validate(upward_state)

        for elem in flatten(H_hat):
            for value in get_debug_values(elem):
                assert value.shape[0] == dbm.batch_size
            assert V in theano.gof.graph.ancestors([elem])
            if Y is not None:
                assert Y in theano.gof.graph.ancestors([elem])

        if Y is not None:
            assert all([elem[-1] is Y for elem in history])
            assert H_hat[-1] is Y

        for elem in history:
            assert len(elem) == len(dbm.hidden_layers)

        if return_history:
            for hist_elem, H_elem in safe_zip(history[-1], H_hat):
                assert hist_elem is H_elem
            return history
        else:
            return H_hat

    def do_inpainting(self, V, Y = None, drop_mask = None, drop_mask_Y = None,
            return_history = False, noise = False, niter = None, block_grad = None):
        """
            Gives the mean field expression for units masked out by drop_mask.
            Uses self.niter mean field updates.

            Comes in two variants, unsupervised and supervised:
                unsupervised:
                    Y and drop_mask_Y are not passed to the method.
                    The method produces V_hat, an inpainted version of V
                supervised:
                    Y and drop_mask_Y are passed to the method.
                    The method produces V_hat and Y_hat

            V: a theano batch in model.input_space
            Y: a theano batch in model.output_space, ie, in the output
                space of the last hidden layer
                (it's not really a hidden layer anymore, but oh well.
                it's convenient to code it this way because the labels
                are sort of "on top" of everything else)
                *** Y is always assumed to be a matrix of one-hot category
                labels. ***
            drop_mask: a theano batch in model.input_space
                Should be all binary, with 1s indicating that the corresponding
                element of X should be "dropped", ie, hidden from the algorithm
                and filled in as part of the inpainting process
            drop_mask_Y: a theano vector
                Since we assume Y is a one-hot matrix, each row is a single
                categorical variable. drop_mask_Y is a binary mask specifying
                which *rows* to drop.
        """

        theano_rng = MRG_RandomStreams(2012 + 11 + 7)

        dbm = self.dbm

        warnings.warn("""Should add unit test that calling this with a batch of
                different inputs should yield the same output for each if noise
                is False and drop_mask is all 1s""")

        if niter is None:
            niter = dbm.niter


        assert drop_mask is not None
        assert return_history in [True, False]
        assert noise in [True, False]
        if Y is None:
            if drop_mask_Y is not None:
                raise ValueError("do_inpainting got drop_mask_Y but not Y.")
        else:
            if drop_mask_Y is None:
                raise ValueError("do_inpainting got Y but not drop_mask_Y.")

        if Y is not None:
            assert isinstance(dbm.hidden_layers[-1], Softmax)
            if drop_mask_Y.ndim != 1:
                raise ValueError("do_inpainting assumes Y is a matrix of one-hot labels,"
                        "so each example is only one variable. drop_mask_Y should "
                        "therefore be a vector, but we got something with ndim " +
                        str(drop_mask_Y.ndim))
            drop_mask_Y = drop_mask_Y.dimshuffle(0, 'x')

        orig_V = V
        orig_drop_mask = drop_mask

        history = []

        V_hat, V_hat_unmasked = dbm.visible_layer.init_inpainting_state(V,drop_mask,noise, return_unmasked = True)
        assert V_hat_unmasked.ndim > 1

        H_hat = [layer.init_mf_state() for layer in dbm.hidden_layers]

        if Y is not None:
            Y_hat_unmasked = dbm.hidden_layers[-1].init_inpainting_state(Y, noise)
            Y_hat = drop_mask_Y * Y_hat_unmasked + (1 - drop_mask_Y) * Y
            H_hat[-1] = Y_hat

        V_dropout = self.V_dropout
        H_dropout = self.H_dropout

        def apply_dropout(x, d):
            if isinstance(x, (list, tuple)):
                return [ apply_dropout(x_elem, d_elem) for x_elem, d_elem in safe_zip(x, d) ]
            return x * d

        def update_history():

            assert V_hat_unmasked.ndim > 1
            d =  { 'V_hat' :  V_hat_not_dropped, 'H_hat' : H_hat_not_dropped, 'V_hat_unmasked' : V_hat_unmasked }
            if Y is not None:
                d['Y_hat_unmasked'] = Y_hat_unmasked
                d['Y_hat'] = Y_hat_not_dropped
            history.append( d )

        V_hat_not_dropped = V_hat
        V_hat = apply_dropout(V_hat, V_dropout)
        if Y is not None:
            Y_hat_not_dropped = H_hat[-1]
        H_hat_not_dropped = list(H_hat)
        H_hat = apply_dropout(H_hat, H_dropout)
        update_history()

        for i in xrange(niter):
            for j in xrange(0, len(H_hat), 2):
                if j == 0:
                    state_below = dbm.visible_layer.upward_state(V_hat)
                else:
                    state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                if j == len(H_hat) - 1:
                    state_above = None
                    layer_above = None
                else:
                    state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                    layer_above = dbm.hidden_layers[j+1]
                H_hat[j] = dbm.hidden_layers[j].mf_update(
                        state_below = state_below,
                        state_above = state_above,
                        layer_above = layer_above)
                if Y is not None and j == len(dbm.hidden_layers) - 1:
                    Y_hat_unmasked = H_hat[j]
                    H_hat[j] = drop_mask_Y * H_hat[j] + (1 - drop_mask_Y) * Y
                    Y_hat_not_dropped = H_hat[-1]
                H_hat_not_dropped[j] = H_hat[j]
                H_hat[j] = apply_dropout(H_hat[j], H_dropout[j])

            V_hat, V_hat_unmasked = dbm.visible_layer.inpaint_update(
                    state_above = dbm.hidden_layers[0].downward_state(H_hat[0]),
                    layer_above = dbm.hidden_layers[0],
                    V = V,
                    drop_mask = drop_mask, return_unmasked = True)
            V_hat_not_dropped = V_hat
            V_hat = apply_dropout(V_hat, V_dropout)
            V_hat.name = 'V_hat[%d](V_hat = %s)' % (i, V_hat.name)


            for j in xrange(1,len(H_hat),2):
                state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                if j == len(H_hat) - 1:
                    state_above = None
                    layer_above = None
                else:
                    state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                    layer_above = dbm.hidden_layers[j+1]
                #end if j
                H_hat[j] = dbm.hidden_layers[j].mf_update(
                        state_below = state_below,
                        state_above = state_above,
                        layer_above = layer_above)
                if Y is not None and j == len(dbm.hidden_layers) - 1:
                    Y_hat_unmasked = H_hat[j]
                    H_hat[j] = drop_mask_Y * H_hat[j] + (1 - drop_mask_Y) * Y
                    Y_hat_not_dropped = H_hat[-1]
                #end if y
                H_hat_not_dropped[j] = H_hat[j]
                H_hat[j] = apply_dropout(H_hat[j], H_dropout[j])
            #end for j
            if block_grad == i + 1:
                V_hat = block_gradient(V_hat)
                V_hat_unmasked = block_gradient(V_hat_unmasked)
                H_hat = block(H_hat)
            update_history()
        #end for i

        # debugging, make sure V didn't get changed in this function
        assert V is orig_V
        assert drop_mask is orig_drop_mask

        Y_hat = H_hat[-1]

        assert V in theano.gof.graph.ancestors([V_hat])
        if Y is not None:
            assert V in theano.gof.graph.ancestors([Y_hat])

        if return_history:
            return history
        else:
            if Y is not None:
                return V_hat_not_dropped, Y_hat_not_dropped
            return V_hat_not_dropped

    def set_batch_size(self, batch_size):
        for var in flatten([self.V_dropout, self.H_dropout]):
            old_val = var.get_value()
            shape = list(old_val.shape)
            shape[0] = batch_size
            new_val = np.ones(shape, dtype=old_val.dtype)/self.include_prob
            var.set_value(new_val)

    def multi_infer(self, V, return_history = False, niter = None, block_grad = None):

        dbm = self.dbm

        assert return_history in [True, False, 0, 1]

        if niter is None:
            niter = dbm.niter

        new_V = 0.5 * V + 0.5 * dbm.visible_layer.init_inpainting_state(V,drop_mask = None,noise = False, return_unmasked = False)

        H_hat = []
        for i in xrange(0,len(dbm.hidden_layers)-1):
            #do double weights update for_layer_i
            if i == 0:
                H_hat.append(dbm.hidden_layers[i].mf_update(
                                                            state_above = None,
                                                            double_weights = True,
                                                            state_below = dbm.visible_layer.upward_state(new_V),
                                                            iter_name = '0'))
            else:
                H_hat.append(dbm.hidden_layers[i].mf_update(
                                                            state_above = None,
                                                            double_weights = True,
                                                            state_below = dbm.hidden_layers[i-1].upward_state(H_hat[i-1]),
                                                            iter_name = '0'))

        #last layer does not need its weights doubled, even on the first pass
        if len(dbm.hidden_layers) > 1:
            H_hat.append(dbm.hidden_layers[-1].mf_update(
                                                         state_above = None,
                                                         state_below = dbm.hidden_layers[-2].upward_state(H_hat[-1])))
        else:
            H_hat.append(dbm.hidden_layers[-1].mf_update(
                                                         state_above = None,
                                                         state_below = dbm.visible_layer.upward_state(V)))

        if block_grad == 1:
            H_hat = block(H_hat)

        history = [ (new_V, list(H_hat)) ]


        #we only need recurrent inference if there are multiple layers
        if len(H_hat) > 1:
            for i in xrange(1, niter):
                for j in xrange(0,len(H_hat),2):
                    if j == 0:
                        state_below = dbm.visible_layer.upward_state(new_V)
                    else:
                        state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                    if j == len(H_hat) - 1:
                        state_above = None
                        layer_above = None
                    else:
                        state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                        layer_above = dbm.hidden_layers[j+1]
                    H_hat[j] = dbm.hidden_layers[j].mf_update(
                                                              state_below = state_below,
                                                              state_above = state_above,
                                                              layer_above = layer_above)
                V_hat = dbm.visible_layer.inpaint_update(
                                                                                 state_above = dbm.hidden_layers[0].downward_state(H_hat[0]),
                                                                                 layer_above = dbm.hidden_layers[0],
                                                                                 V = V,
                                                                                 drop_mask = None)
                new_V = 0.5 * V_hat + 0.5 * V

                for j in xrange(1,len(H_hat),2):
                    state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                    if j == len(H_hat) - 1:
                        state_above = None
                        state_above = None
                    else:
                        state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                        layer_above = dbm.hidden_layers[j+1]
                    H_hat[j] = dbm.hidden_layers[j].mf_update(
                                                              state_below = state_below,
                                                              state_above = state_above,
                                                              layer_above = layer_above)
                #end ifelse
                #end for odd layer

                if block_grad == i:
                    H_hat = block(H_hat)
                    V_hat = block_gradient(V_hat)

                history.append((new_V, list(H_hat)))
        # end for mf iter
        # end if recurrent
        # Run some checks on the output
        for layer, state in safe_izip(dbm.hidden_layers, H_hat):
            upward_state = layer.upward_state(state)
            layer.get_output_space().validate(upward_state)

        inferred = H_hat
        for elem in flatten(inferred):
            for value in get_debug_values(elem):
                assert value.shape[0] == dbm.batch_size
            assert V in gof.graph.ancestors([elem])

        if return_history:
            return history
        else:
            return H_hat[-1]

def mask_weights(input_shape,
                stride,
                shape,
                channels):
    """
        input_shape: how to view a vector below as (rows, cols, channels)
        stride: (row stride, col_stride) between receptive fields on this layer
        channels: how many units should have the same receptive field on this layer
    """

    ipt = np.zeros(input_shape)
    r, c, ch = ipt.shape
    dim = r * c * ch

    mask = []

    out_rows = 0
    for i in xrange(0, input_shape[0] - shape[0] + 1, stride[0]):
        out_rows += 1
        out_cols = 0
        for j in xrange(0, input_shape[1] - shape[1] + 1, stride[1]):
            out_cols += 1
            cur_ipt = ipt.copy()
            cur_ipt[i:i+shape[0], j:j+shape[1], :] = 1.
            cur_mask = cur_ipt.reshape(dim, 1)
            mask.extend([ cur_mask ] * channels)

    print 'masked hidden shape: ', (out_rows, out_cols)

    return np.concatenate(mask, axis=1)

class ProductDecay(Cost):

    def __init__(self, coeff):
        self.coeff = coeff

    def __call__(self, model, X, Y=None, **kwargs):
        h1, h2 = model.hidden_layers[0:2]
        W1, = h1.transformer.get_params()
        W2, = h2.transformer.get_params()
        prod = T.dot(W1, W2)
        fro = T.sqr(prod).sum()
        return self.coeff * fro


class Recons(Cost):

    def __init__(self, supervised, coeffs):
        self.__dict__.update(locals())
        del self.self

    def __call__(self, model, X, Y = None, ** kwargs):

        if not self.supervised:
            raise NotImplementedError()
        else:
            Q = model.mf(X, Y = Y, niter = model.niter / 2)
            assert len(Q) == 3
            assert len(model.hidden_layers) == 3
            assert isinstance(model.hidden_layers[0], dbm.BinaryVectorMaxPool)
            assert isinstance(model.hidden_layers[1], dbm.BinaryVectorMaxPool)
            assert model.hidden_layers[0].pool_size == 1
            assert model.hidden_layers[1].pool_size == 1

            H1, H2 = Q[0:2]
            H1, _ = H1
            H2, _ = H2

            h1, h2, y = model.hidden_layers
            v = model.visible_layer

            # H1 penalty
            V = T.nnet.sigmoid(h1.downward_message(H1)+v.bias)
            V_recons = v.recons_cost(X, V, T.zeros_like(X))

            H2new = H2

            for i in xrange(model.niter / 2):
                Y_hat = y.mf_update(state_below = H2new)
                H2new = h2.mf_update(state_below = H1, layer_above = y, state_above = Y_hat)[0]
            Y_recons = y.recons_cost(Y, Y_hat, T.zeros_like(Y[:,0]), 1./T.cast(X.shape[1], 'float32'))

            total_cost = self.coeffs[0] * (V_recons + Y_recons)

            # H2 penalty

            Y_hat = y.mf_update(state_below = H2)
            total_cost += self.coeffs[1] * y.recons_cost(Y, Y_hat,
                    T.zeros_like(Y[:,0]), 1./T.cast(X.shape[1], 'float32'))

            for i in xrange(model.niter / 2):
                V_hat = T.nnet.sigmoid(h1.downward_message(H1)+v.bias)
                H1 = h1.mf_update(state_below = V_hat, state_above = H2, layer_above = h2)[0]

            total_cost += self.coeffs[1] * v.recons_cost(X, V_hat, T.zeros_like(X))

            return total_cost

from pylearn2.models.dbm import BVMP_Gaussian

def freeze_layer_0(super_dbm):
    super_dbm.freeze(super_dbm.hidden_layers[0].get_params())
    return super_dbm

class SpeedMonitoringDBM(SuperDBM):

    def __init__(self, ** kwargs):
        SuperDBM.__init__(self, ** kwargs)
        self.param_speed = sharedX(0.)

    def censor_updates(self, updates):

        SuperDBM.censor_updates(self, updates)

        cur_param_speed = 0.

        for param in self.get_params():
            cur_param_speed += T.sqr(param - updates[param]).sum()

        cur_param_speed = T.sqrt(cur_param_speed)

        time_constant = .01

        updates[self.param_speed]  = (1. - time_constant) * self.param_speed + time_constant * cur_param_speed


    def get_monitoring_channels(self, X, Y=None, **kwargs):

        rval = SuperDBM.get_monitoring_channels(self, X, Y, ** kwargs)

        rval['param_speed'] = self.param_speed

        return rval


class SoftplusMaxPool(HiddenLayer):
    """
        A hidden layer that does max-pooling on  a sum of infinitely many binary vectors.
        See Nair and Hinton's paper on rectified linear RBMs. This can be approximated well
        with softplus which is what we do here.
        It has two sublayers, the detector layer and the pooling
        layer. The detector layer is its downward state and the pooling
        layer is its upward state.

        TODO: this layer uses (pooled, detector) as its total state,
              which can be confusing when listing all the states in
              the network left to right. Change this and
              pylearn2.expr.probabilistic_max_pooling to use
              (detector, pooled)
    """

    def __init__(self,
             detector_layer_dim,
            pool_size,
            layer_name,
            max_act = None,
            irange = None,
            sparse_init = None,
            sparse_stdev = 1.,
            include_prob = 1.0,
            init_bias = 0.,
            W_lr_scale = None,
            b_lr_scale = None,
            center = False,
            mask_weights = None,
            max_col_norm = None,
            beta_bias_layer = None,
            copies = 1):
        """

            include_prob: probability of including a weight element in the set
                    of weights initialized to U(-irange, irange). If not included
                    it is initialized to 0.

        """
        self.__dict__.update(locals())
        del self.self

        self._b = sharedX( np.zeros((self.detector_layer_dim,)) + init_bias, name = layer_name + '_b')

        if self.center:
            if self.pool_size != 1:
                raise NotImplementedError()
            self.offset = sharedX(self.nonlinearity(self.bias()).eval())

    def nonlinearity(self, x):
        if self.max_act:
            return T.nnet.softplus(x) - T.nnet.softplus(x-self.max_act)
        return T.nnet.softplus(x)

    def bias(self):

        if self.beta_bias_layer:
            W, = self.transformer.get_params()
            beta = self.beta_bias_layer.beta
            assert beta.ndim == 1
            return self._b - 0.5 * T.dot(beta, T.sqr(W))
        return self._b


    def get_lr_scalers(self):

        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            W, = self.transformer.get_params()
            rval[W] = self.W_lr_scale

        if self.b_lr_scale is not None:
            rval[self._b] = self.b_lr_scale

        return rval

    def set_input_space(self, space):
        """ Note: this resets parameters! """

        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)


        if not (self.detector_layer_dim % self.pool_size == 0):
            raise ValueError("detector_layer_dim = %d, pool_size = %d. Should be divisible but remainder is %d" %
                    (self.detector_layer_dim, self.pool_size, self.detector_layer_dim % self.pool_size))

        self.h_space = VectorSpace(self.detector_layer_dim)
        self.pool_layer_dim = self.detector_layer_dim / self.pool_size
        self.output_space = VectorSpace(self.pool_layer_dim)

        rng = self.dbm.rng
        if self.irange is not None:
            assert self.sparse_init is None
            W = rng.uniform(-self.irange,
                                 self.irange,
                                 (self.input_dim, self.detector_layer_dim)) * \
                    (rng.uniform(0.,1., (self.input_dim, self.detector_layer_dim))
                     < self.include_prob)
        else:
            assert self.sparse_init is not None
            W = np.zeros((self.input_dim, self.detector_layer_dim))
            def mask_rejects(idx, i):
                if self.mask_weights is None:
                    return False
                return self.mask_weights[idx, i] == 0.
            for i in xrange(self.detector_layer_dim):
                assert self.sparse_init <= self.input_dim
                for j in xrange(self.sparse_init):
                    idx = rng.randint(0, self.input_dim)
                    while W[idx, i] != 0 or mask_rejects(idx, i):
                        idx = rng.randint(0, self.input_dim)
                    W[idx, i] = rng.randn()
            W *= self.sparse_stdev

        W = sharedX(W)
        W.name = self.layer_name + '_W'

        self.transformer = MatrixMul(W)

        W ,= self.transformer.get_params()
        assert W.name is not None

        if self.mask_weights is not None:
            expected_shape =  (self.input_dim, self.detector_layer_dim)
            if expected_shape != self.mask_weights.shape:
                raise ValueError("Expected mask with shape "+str(expected_shape)+" but got "+str(self.mask_weights.shape))
            self.mask = sharedX(self.mask_weights)

    def censor_updates(self, updates):

        # Patch old pickle files
        if not hasattr(self, 'mask_weights'):
            self.mask_weights = None
        if not hasattr(self, 'max_col_norm'):
            self.max_col_norm = None

        if self.mask_weights is not None:
            W ,= self.transformer.get_params()
            if W in updates:
                updates[W] = updates[W] * self.mask

        if self.max_col_norm is not None:
            W, = self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))


    def get_total_state_space(self):
        return CompositeSpace((self.output_space, self.h_space))

    def get_params(self):
        assert self._b.name is not None
        W ,= self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self._b not in rval
        rval.append(self._b)
        return rval

    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W ,= self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    def get_weights(self):
        if self.requires_reformat:
            # This is not really an unimplemented case.
            # We actually don't know how to format the weights
            # in design space. We got the data in topo space
            # and we don't have access to the dataset
            raise NotImplementedError()
        W ,= self.transformer.get_params()
        W = W.get_value()

        if self.beta_bias_layer:
            x = raw_input("multiply by beta?")
            if x == 'y':
                beta = self.beta_bias_layer.beta.get_value()
                return (W.T * beta).T
            assert x == 'n'
        return W

    def set_weights(self, weights):
        W, = self.transformer.get_params()
        W.set_value(weights)

    def set_biases(self, biases, recenter = False):
        self._b.set_value(biases)
        if recenter:
            assert self.center
            if self.pool_size != 1:
                raise NotImplementedError()
            self.offset.set_value(sigmoid_numpy(self.b.get_value()))

    def get_biases(self):
        return self._b.get_value()

    def get_weights_format(self):
        return ('v', 'h')

    def get_weights_view_shape(self):
        total = self.detector_layer_dim
        cols = self.pool_size
        if cols == 1:
            # Let the PatchViewer decidew how to arrange the units
            # when they're not pooled
            raise NotImplementedError()
        # When they are pooled, make each pooling unit have one row
        rows = total / cols
        return rows, cols


    def get_weights_topo(self):

        if not isinstance(self.input_space, Conv2DSpace):
            raise NotImplementedError()

        W ,= self.transformer.get_params()

        W = W.T

        W = W.reshape((self.detector_layer_dim, self.input_space.shape[0],
            self.input_space.shape[1], self.input_space.nchannels))

        W = Conv2DSpace.convert(W, self.input_space.axes, ('b', 0, 1, 'c'))

        return function([], W)()

    def upward_state(self, total_state):
        p,h = total_state
        self.h_space.validate(h)
        self.output_space.validate(p)

        if not hasattr(self, 'center'):
            self.center = False

        if self.center:
            return p - self.offset

        if not hasattr(self, 'copies'):
            self.copies = 1

        return p * self.copies

    def downward_state(self, total_state):
        p,h = total_state

        if not hasattr(self, 'center'):
            self.center = False

        if self.center:
            return h - self.offset

        return h * self.copies

    def get_monitoring_channels(self):

        W ,= self.transformer.get_params()

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        return OrderedDict([
              ('row_norms_min'  , row_norms.min()),
              ('row_norms_mean' , row_norms.mean()),
              ('row_norms_max'  , row_norms.max()),
              ('col_norms_min'  , col_norms.min()),
              ('col_norms_mean' , col_norms.mean()),
              ('col_norms_max'  , col_norms.max()),
            ])


    def get_monitoring_channels_from_state(self, state):

        P, H = state

        rval = OrderedDict()

        if self.pool_size == 1:
            vars_and_prefixes = [ (P,'') ]
        else:
            vars_and_prefixes = [ (P, 'p_'), (H, 'h_') ]

        for var, prefix in vars_and_prefixes:
            v_max = var.max(axis=0)
            v_min = var.min(axis=0)
            v_mean = var.mean(axis=0)
            v_range = v_max - v_min

            # max_x.mean_u is "the mean over *u*nits of the max over e*x*amples"
            # The x and u are included in the name because otherwise its hard
            # to remember which axis is which when reading the monitor
            # I use inner.outer rather than outer_of_inner or something like that
            # because I want mean_x.* to appear next to each other in the alphabetical
            # list, as these are commonly plotted together
            for key, val in [
                    ('max_x.max_u', v_max.max()),
                    ('max_x.mean_u', v_max.mean()),
                    ('max_x.min_u', v_max.min()),
                    ('min_x.max_u', v_min.max()),
                    ('min_x.mean_u', v_min.mean()),
                    ('min_x.min_u', v_min.min()),
                    ('range_x.max_u', v_range.max()),
                    ('range_x.mean_u', v_range.mean()),
                    ('range_x.min_u', v_range.min()),
                    ('mean_x.max_u', v_mean.max()),
                    ('mean_x.mean_u', v_mean.mean()),
                    ('mean_x.min_u', v_mean.min())
                    ]:
                rval[prefix+key] = val

        return rval

    def get_stdev_rewards(self, state, coeffs):
        rval = 0.

        P, H = state
        self.output_space.validate(P)
        self.h_space.validate(H)


        if self.pool_size == 1:
            # If the pool size is 1 then pools = detectors
            # and we should not penalize pools and detectors separately
            assert len(state) == 2
            if isinstance(coeffs, str):
                coeffs = float(coeffs)
            assert isinstance(coeffs, float)
            _, state = state
            state = [state]
            coeffs = [coeffs]
        else:
            assert all([len(elem) == 2 for elem in [state, coeffs]])

        for s, c in safe_zip(state, coeffs):
            assert all([isinstance(elem, float) for elem in [c]])
            if c == 0.:
                continue
            mn = s.mean(axis=0)
            dev = s - mn
            stdev = T.sqrt(T.sqr(dev).mean(axis=0))
            rval += (0.5 - stdev).mean()*c

        return rval
    def get_range_rewards(self, state, coeffs):
        rval = 0.

        P, H = state
        self.output_space.validate(P)
        self.h_space.validate(H)


        if self.pool_size == 1:
            # If the pool size is 1 then pools = detectors
            # and we should not penalize pools and detectors separately
            assert len(state) == 2
            if isinstance(coeffs, str):
                coeffs = float(coeffs)
            assert isinstance(coeffs, float)
            _, state = state
            state = [state]
            coeffs = [coeffs]
        else:
            assert all([len(elem) == 2 for elem in [state, coeffs]])

        for s, c in safe_zip(state, coeffs):
            assert all([isinstance(elem, float) for elem in [c]])
            if c == 0.:
                continue
            mx = s.max(axis=0)
            assert hasattr(mx.owner.op, 'grad')
            assert mx.ndim == 1
            mn = s.min(axis=0)
            assert hasattr(mn.owner.op, 'grad')
            assert mn.ndim == 1
            r = mx - mn
            rval += (1 - r).mean()*c

        return rval

    def get_l1_act_cost(self, state, target, coeff, eps = None):
        rval = 0.

        P, H = state
        self.output_space.validate(P)
        self.h_space.validate(H)


        if self.pool_size == 1:
            # If the pool size is 1 then pools = detectors
            # and we should not penalize pools and detectors separately
            assert len(state) == 2
            if not isinstance(target, float):
                raise TypeError("BinaryVectorMaxPool.get_l1_act_cost expected target of type float " + \
                        " but an instance named "+self.layer_name + " got target "+str(target) + " of type "+str(type(target)))
            assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
            _, state = state
            state = [state]
            target = [target]
            coeff = [coeff]
            if eps is None:
                eps = [0.]
            else:
                eps = [eps]
        else:
            assert all([len(elem) == 2 for elem in [state, target, coeff]])
            if eps is None:
                eps = [0., 0.]
            if target[1] > target[0]:
                warnings.warn("Do you really want to regularize the detector units to be more active than the pooling units?")

        for s, t, c, e in safe_zip(state, target, coeff, eps):
            assert all([isinstance(elem, float) or hasattr(elem, 'dtype') for elem in [t, c, e]])
            if c == 0.:
                continue
            m = s.mean(axis=0)
            assert m.ndim == 1
            rval += T.maximum(abs(m-t)-e,0.).mean()*c

        return rval

    def get_l2_act_cost(self, state, target, coeff):
        rval = 0.

        P, H = state
        self.output_space.validate(P)
        self.h_space.validate(H)


        if self.pool_size == 1:
            # If the pool size is 1 then pools = detectors
            # and we should not penalize pools and detectors separately
            assert len(state) == 2
            if not isinstance(target, float):
                raise TypeError("BinaryVectorMaxPool.get_l1_act_cost expected target of type float " + \
                        " but an instance named "+self.layer_name + " got target "+str(target) + " of type "+str(type(target)))
            assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
            _, state = state
            state = [state]
            target = [target]
            coeff = [coeff]
        else:
            assert all([len(elem) == 2 for elem in [state, target, coeff]])
            if target[1] > target[0]:
                warnings.warn("Do you really want to regularize the detector units to be more active than the pooling units?")

        for s, t, c in safe_zip(state, target, coeff):
            assert all([isinstance(elem, float) or hasattr(elem, 'dtype') for elem in [t, c]])
            if c == 0.:
                continue
            m = s.mean(axis=0)
            assert m.ndim == 1
            rval += T.square(m-t).mean()*c

        return rval

    def sample(self, state_below = None, state_above = None,
            layer_above = None,
            theano_rng = None):
        raise NotImplementedError("Just copied from BVMP")
        if self.copies != 1:
            raise NotImplementedError()

        if theano_rng is None:
            raise ValueError("theano_rng is required; it just defaults to None so that it may appear after layer_above / state_above in the list.")

        if state_above is not None:
            msg = layer_above.downward_message(state_above)
        else:
            msg = None

        if self.requires_reformat:
            state_below = self.input_space.format_as(state_below, self.desired_space)

        z = self.transformer.lmul(state_below) + self.b
        p, h, p_sample, h_sample = max_pool_channels(z,
                self.pool_size, msg, theano_rng)

        return p_sample, h_sample

    def downward_message(self, downward_state):
        self.h_space.validate(downward_state)
        rval = self.transformer.lmul_T(downward_state)

        if self.requires_reformat:
            rval = self.desired_space.format_as(rval, self.input_space)

        return rval * self.copies

    def init_mf_state(self):
        # work around theano bug with broadcasted vectors
        z = T.alloc(0., self.dbm.batch_size, self.detector_layer_dim).astype(self._b.dtype) + \
                self.bias().dimshuffle('x', 0)
        rval = self.max_pool_channels(z = z,
                pool_size = self.pool_size)
        return rval

    def make_state(self, num_examples, numpy_rng):
        """ Returns a shared variable containing an actual state
           (not a mean field state) for this variable.
        """
        raise NotImplementedError("Just copied from BVMP")

        if not hasattr(self, 'copies'):
            self.copies = 1

        if self.copies != 1:
            raise NotImplementedError()


        empty_input = self.h_space.get_origin_batch(num_examples)
        empty_output = self.output_space.get_origin_batch(num_examples)

        h_state = sharedX(empty_input)
        p_state = sharedX(empty_output)

        theano_rng = MRG_RandomStreams(numpy_rng.randint(2 ** 16))

        default_z = T.zeros_like(h_state) + self.b

        p_exp, h_exp, p_sample, h_sample = max_pool_channels(
                z = default_z,
                pool_size = self.pool_size,
                theano_rng = theano_rng)

        assert h_sample.dtype == default_z.dtype

        f = function([], updates = [
            (p_state , p_sample),
            (h_state , h_sample)
            ])

        f()

        p_state.name = 'p_sample_shared'
        h_state.name = 'h_sample_shared'

        return p_state, h_state

    def expected_energy_term(self, state, average, state_below, average_below):
        raise NotImplementedError("Just copied from BVMP")

        # Don't need to do anything special for centering, upward_state / downward state
        # make it all just work

        self.input_space.validate(state_below)

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.dbm.batch_size is %d but got shape of %d" % (self.dbm.batch_size, sb.shape[0]))
                    assert reduce(lambda x,y: x * y, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below, self.desired_space)

        downward_state = self.downward_state(state)
        self.h_space.validate(downward_state)

        # Energy function is linear so it doesn't matter if we're averaging or not
        # Specifically, our terms are -u^T W d - b^T d where u is the upward state of layer below
        # and d is the downward state of this layer

        bias_term = T.dot(downward_state, self.b)
        weights_term = (self.transformer.lmul(state_below) * downward_state).sum(axis=1)

        rval = -bias_term - weights_term

        assert rval.ndim == 1

        return rval * self.copies

    def linear_feed_forward_approximation(self, state_below):
        raise NotImplementedError("Just copied from BVMP")
        """
        Used to implement TorontoSparsity. Unclear exactly what properties of it are
        important or how to implement it for other layers.

        Properties it must have:
            output is same kind of data structure (ie, tuple of theano 2-tensors)
            as mf_update

        Properties it probably should have for other layer types:
            An infinitesimal change in state_below or the parameters should cause the same sign of change
            in the output of linear_feed_forward_approximation and in mf_update

            Should not have any non-linearities that cause the gradient to shrink

            Should disregard top-down feedback
        """

        z = self.transformer.lmul(state_below) + self.b

        if self.pool_size != 1:
            # Should probably implement sum pooling for the non-pooled version,
            # but in reality it's not totally clear what the right answer is
            raise NotImplementedError()

        return z, z

    def mf_update(self, state_below, state_above, layer_above = None, double_weights = False, iter_name = None):

        self.input_space.validate(state_below)

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.dbm.batch_size is %d but got shape of %d" % (self.dbm.batch_size, sb.shape[0]))
                    assert reduce(lambda x,y: x * y, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below, self.desired_space)

        if iter_name is None:
            iter_name = 'anon'

        if state_above is not None:
            assert layer_above is not None
            msg = layer_above.downward_message(state_above)
            msg.name = 'msg_from_'+layer_above.layer_name+'_to_'+self.layer_name+'['+iter_name+']'
        else:
            msg = None

        if double_weights:
            state_below = 2. * state_below
            state_below.name = self.layer_name + '_'+iter_name + '_2state'
        z = self.transformer.lmul(state_below) + self.bias()
        if self.layer_name is not None and iter_name is not None:
            z.name = self.layer_name + '_' + iter_name + '_z'
        p,h = self.max_pool_channels(z, self.pool_size, msg)

        p.name = self.layer_name + '_p_' + iter_name
        h.name = self.layer_name + '_h_' + iter_name

        return p, h

    def max_pool_channels(self, z, pool_size, top_down = None):

        if pool_size == 1:
            if top_down is None:
                top_down = 0.
            total_input = z + top_down
            p = self.nonlinearity(total_input)
            h = p
        else:
            raise NotImplementedError()

        return p, h

from pylearn2.models.dbm import BiasInit
from pylearn2.models.dbm import MoreConsistent
from pylearn2.models.dbm import MoreConsistent2
