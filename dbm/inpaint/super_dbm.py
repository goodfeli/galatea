from pylearn2.models.model import Model
import theano
from pylearn2.space import Space
from pylearn2.space import Conv2DSpace
from pylearn2.space import VectorSpace
from pylearn2.space import CompositeSpace
from pylearn2.utils import sharedX
from pylearn2.linear.conv2d import make_random_conv2D
import theano.tensor as T
import numpy as np
from pylearn2.expr.probabilistic_max_pooling import max_pool
from pylearn2.expr.probabilistic_max_pooling import max_pool_b01c
from theano.printing import Print
from galatea.theano_upgrades import block_gradient
import warnings
from theano import function
from theano.sandbox.rng_mrg import MRG_RandomStreams
import time
from pylearn2.costs.cost import Cost
from pylearn2.utils import safe_zip
from pylearn2.utils import safe_izip
from galatea.theano_upgrades import _ElemwiseNoGradient
from theano import config
io = None
from pylearn2.training_callbacks.training_callback import TrainingCallback
from pylearn2.models.dbm import BinaryVectorMaxPool
from pylearn2.models.dbm import DBM
from pylearn2.models.dbm import Layer
from pylearn2.models.dbm import HiddenLayer
from pylearn2.models import dbm

def flatten(l):
    rval = []
    for elem in l:
        if isinstance(elem, (list, tuple)):
            rval.extend(flatten(elem))
        else:
            rval.append(elem)
    return rval
def block(l):
    new = []
    for elem in l:
        if isinstance(elem, (list, tuple)):
            new.append(block(elem))
        else:
            new.append(block_gradient(elem))
    if isinstance(l, tuple):
        return tuple(new)
    return new

class SuperDBM(DBM):

    # Constructor is handled by DBM superclass

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
        if isinstance(self.hidden_layers[-1], Softmax):
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

    def expected_energy(self, V, mf_hidden):
        """
            V: a theano batch of visible unit observations
                (must be SAMPLES, not mean field parameters:
                    the random variables in the expectation are
                    the hiddens only)

            mf_hidden: a list, one element per hidden layer, of
                      batches of variational parameters
                (must be VARIATIONAL PARAMETERS, not samples.
                Layers with analytically determined variance parameters
                for their mean field parameters will use those to integrate
                over the variational distribution, so it's not generally
                the same thing as measuring the energy at a point.)

            returns: a vector containing the expected energy of
                    each example under the corresponding variational
                    distribution.
        """

        self.visible_layer.space.validate(V)
        assert isinstance(mf_hidden, (list, tuple))
        assert len(mf_hidden) == len(self.hidden_layers)

        terms = []

        terms.append(self.visible_layer.expected_energy_term(state = V, average=False))

        assert len(self.hidden_layers) > 0 # this could be relaxed, but current code assumes it

        terms.append(self.hidden_layers[0].expected_energy_term(
            state_below=self.visible_layer.upward_state(V), average_below=False,
            state=mf_hidden[0], average=True))

        for i in xrange(1, len(self.hidden_layers)):
            layer = self.hidden_layers[i]
            layer_below = self.hidden_layers[i-1]
            mf_below = mf_hidden[i-1]
            mf_below = layer_below.upward_state(mf_below)
            mf = mf_hidden[i]
            terms.append(layer.expected_energy_term(state_below=mf_below, state=mf,
                average_below=True, average=True))

        assert len(terms) > 0

        rval = reduce(lambda x, y: x + y, terms)

        assert rval.ndim == 1
        return rval


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

        warnings.warn("""Should add unit test that calling this with a batch of
                different inputs should yield the same output for each if noise
                is False and drop_mask is all 1s""")

        if niter is None:
            niter = self.niter


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
            assert isinstance(self.hidden_layers[-1], Softmax)
            if drop_mask_Y.ndim != 1:
                raise ValueError("do_inpainting assumes Y is a matrix of one-hot labels,"
                        "so each example is only one variable. drop_mask_Y should "
                        "therefore be a vector, but we got something with ndim " +
                        str(drop_mask_Y.ndim))
            drop_mask_Y = drop_mask_Y.dimshuffle(0, 'x')

        orig_V = V
        orig_drop_mask = drop_mask

        history = []

        V_hat, V_hat_unmasked = self.visible_layer.init_inpainting_state(V,drop_mask,noise, return_unmasked = True)
        assert V_hat_unmasked.ndim > 1

        H_hat = []
        for i in xrange(0,len(self.hidden_layers)-1):
            #do double weights update for_layer_i
            if i == 0:
                H_hat.append(self.hidden_layers[i].mf_update(
                    state_above = None,
                    double_weights = True,
                    state_below = self.visible_layer.upward_state(V_hat),
                    iter_name = '0'))
            else:
                H_hat.append(self.hidden_layers[i].mf_update(
                    state_above = None,
                    double_weights = True,
                    state_below = self.hidden_layers[i-1].upward_state(H_hat[i-1]),
                    iter_name = '0'))
        #last layer does not need its weights doubled, even on the first pass
        if len(self.hidden_layers) > 1:
            H_hat.append( self.hidden_layers[-1].mf_update(
                state_above = None,
                #layer_above = None,
                state_below = self.hidden_layers[-2].upward_state(H_hat[-1])))
        else:
            H_hat.append( self.hidden_layers[-1].mf_update(
                state_above = None,
                state_below = self.visible_layer.upward_state(V_hat)))

        if Y is not None:
            Y_hat_unmasked = self.hidden_layers[-1].init_inpainting_state(Y, noise)
            Y_hat = drop_mask_Y * Y_hat_unmasked + (1 - drop_mask_Y) * Y
            H_hat[-1] = Y_hat
            if len(self.hidden_layers) > 1:
                i = len(self.hidden_layers) - 2
                if i == 0:
                    H_hat[i] = self.hidden_layers[i].mf_update(
                        state_above = Y_hat,
                        layer_above = self.hidden_layers[-1],
                        state_below = self.visible_layer.upward_state(V_hat),
                        iter_name = '0')
                else:
                    H_hat[i] = self.hidden_layers[i].mf_update(
                        state_above = Y_hat,
                        layer_above = self.hidden_layers[-1],
                        state_below = self.hidden_layers[i-1].upward_state(H_hat[i-1]),
                        iter_name = '0')


        def update_history():
            assert V_hat_unmasked.ndim > 1
            d =  { 'V_hat' :  V_hat, 'H_hat' : H_hat, 'V_hat_unmasked' : V_hat_unmasked }
            if Y is not None:
                d['Y_hat_unmasked'] = Y_hat_unmasked
                d['Y_hat'] = H_hat[-1]
            history.append( d )

        if block_grad == 1:
            V_hat = block_gradient(V_hat)
            V_hat_unmasked = block_gradient(V_hat_unmasked)
            H_hat = block(H_hat)
        update_history()

        for i in xrange(niter-1):
            for j in xrange(0, len(H_hat), 2):
                if j == 0:
                    state_below = self.visible_layer.upward_state(V_hat)
                else:
                    state_below = self.hidden_layers[j-1].upward_state(H_hat[j-1])
                if j == len(H_hat) - 1:
                    state_above = None
                    layer_above = None
                else:
                    state_above = self.hidden_layers[j+1].downward_state(H_hat[j+1])
                    layer_above = self.hidden_layers[j+1]
                H_hat[j] = self.hidden_layers[j].mf_update(
                        state_below = state_below,
                        state_above = state_above,
                        layer_above = layer_above)
                if Y is not None and j == len(self.hidden_layers) - 1:
                    Y_hat_unmasked = H_hat[j]
                    H_hat[j] = drop_mask_Y * H_hat[j] + (1 - drop_mask_Y) * Y

            V_hat, V_hat_unmasked = self.visible_layer.inpaint_update(
                    state_above = self.hidden_layers[0].downward_state(H_hat[0]),
                    layer_above = self.hidden_layers[0],
                    V = V,
                    drop_mask = drop_mask, return_unmasked = True)
            V_hat.name = 'V_hat[%d](V_hat = %s)' % (i, V_hat.name)

            for j in xrange(1,len(H_hat),2):
                state_below = self.hidden_layers[j-1].upward_state(H_hat[j-1])
                if j == len(H_hat) - 1:
                    state_above = None
                    layer_above = None
                else:
                    state_above = self.hidden_layers[j+1].downward_state(H_hat[j+1])
                    layer_above = self.hidden_layers[j+1]
                #end if j
                H_hat[j] = self.hidden_layers[j].mf_update(
                        state_below = state_below,
                        state_above = state_above,
                        layer_above = layer_above)
                if Y is not None and j == len(self.hidden_layers) - 1:
                    Y_hat_unmasked = H_hat[j]
                    H_hat[j] = drop_mask_Y * H_hat[j] + (1 - drop_mask_Y) * Y
                #end if y
            #end for j
            if block_grad == i:
                V_hat = block_gradient(V_hat)
                V_hat_unmasked = block_gradient(V_hat_unmasked)
                H_hat = block(H_hat)
            update_history()
        #end for i

        # debugging, make sure V didn't get changed in this function
        assert V is orig_V
        assert drop_mask is orig_drop_mask

        if return_history:
            return history
        else:
            return V_hat

    def mf(self, V, Y = None, return_history = False, niter = None, block_grad = None):


        assert Y not in [True, False, 0, 1]
        assert return_history in [True, False, 0, 1]

        if Y is not None:
            self.hidden_layers[-1].get_output_space().validate(Y)


        if niter is None:
            niter = self.niter

        H_hat = []
        for i in xrange(0,len(self.hidden_layers)-1):
            #do double weights update for_layer_i
            if i == 0:
                H_hat.append(self.hidden_layers[i].mf_update(
                    state_above = None,
                    double_weights = True,
                    state_below = self.visible_layer.upward_state(V),
                    iter_name = '0'))
            else:
                H_hat.append(self.hidden_layers[i].mf_update(
                    state_above = None,
                    double_weights = True,
                    state_below = self.hidden_layers[i-1].upward_state(H_hat[i-1]),
                    iter_name = '0'))

        #last layer does not need its weights doubled, even on the first pass
        if len(self.hidden_layers) > 1:
            H_hat.append( self.hidden_layers[-1].mf_update(
                state_above = None,
                state_below = self.hidden_layers[-2].upward_state(H_hat[-1])))
        else:
            H_hat.append( self.hidden_layers[-1].mf_update(
                state_above = None,
                state_below = self.visible_layer.upward_state(V)))

        # Make corrections for if we're also running inference on Y
        if Y is not None:
            state_above = self.hidden_layers[-1].downward_state(Y)
            layer_above = self.hidden_layers[-1]
            assert len(self.hidden_layers) > 1

            # Last layer before Y does not need its weights doubled
            # because it already has top down input
            if len(self.hidden_layers) > 2:
                state_below = self.hidden_layers[-3].upward_state(H_hat[-3])
            else:
                state_below = self.visible_layer.upward_state(V)

            H_hat[-2] = self.hidden_layers[-2].mf_update(
                            state_below = state_below,
                            state_above = state_above,
                            layer_above = layer_above)

            # Last layer is clamped to Y
            H_hat[-1] = Y



        if block_gradient == 1:
            H_hat = block(H_hat)

        history = [ list(H_hat) ]


        #we only need recurrent inference if there are multiple layers
        if len(H_hat) > 1:
            for i in xrange(niter-1):
                for j in xrange(0,len(H_hat),2):
                    if j == 0:
                        state_below = self.visible_layer.upward_state(V)
                    else:
                        state_below = self.hidden_layers[j-1].upward_state(H_hat[j-1])
                    if j == len(H_hat) - 1:
                        state_above = None
                        layer_above = None
                    else:
                        state_above = self.hidden_layers[j+1].downward_state(H_hat[j+1])
                        layer_above = self.hidden_layers[j+1]
                    H_hat[j] = self.hidden_layers[j].mf_update(
                            state_below = state_below,
                            state_above = state_above,
                            layer_above = layer_above)

                if Y is not None:
                    H_hat[-1] = Y

                for j in xrange(1,len(H_hat),2):
                    state_below = self.hidden_layers[j-1].upward_state(H_hat[j-1])
                    if j == len(H_hat) - 1:
                        state_above = None
                        state_above = None
                    else:
                        state_above = self.hidden_layers[j+1].downward_state(H_hat[j+1])
                        layer_above = self.hidden_layers[j+1]
                    H_hat[j] = self.hidden_layers[j].mf_update(
                            state_below = state_below,
                            state_above = state_above,
                            layer_above = layer_above)
                    #end ifelse
                #end for odd layer

                if Y is not None:
                    H_hat[-1] = Y

                if block_grad == i:
                    H_hat = block(H_hat)

                history.append(list(H_hat))
            # end for mf iter
        # end if recurrent

        # Run some checks on the output
        for layer, state in safe_izip(self.hidden_layers, H_hat):
            upward_state = layer.upward_state(state)
            layer.get_output_space().validate(upward_state)
        if Y is not None:
            assert all([elem[-1] is Y for elem in history])
            assert H_hat[-1] is Y

        if return_history:
            return history
        else:
            return H_hat

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

        layer_to_rao_blackwellized = {}

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



    def reconstruct(self, V):

        H = self.mf(V)[0]

        downward_state = self.hidden_layers[0].downward_state(H)

        recons = self.visible_layer.inpaint_update(
                layer_above = self.hidden_layers[0],
                state_above = downward_state,
                drop_mask = None, V = None)

        return recons

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
            #hid_stuff = Print('hid_stuff',attrs=['min'])(hid_stuff)

            # this should be non-negative
            vis_stuff =  hid.transformer.lmul_sq_T(hid_stuff)
            #vis_stuff = Print('vis_stuff',attrs=['min'])(vis_stuff)

            sq_beta = T.sqr(beta)

            # this should be non-negative
            first_term_presum = sq_beta *(0.5* T.square(V-recons)+vis_stuff)
            #first_term_presum = Print('first_term_presum',attrs=['min'])(first_term_presum)
            first_term = first_term_presum.sum(axis=(1,2,3)).mean()
            assert first_term.ndim == 0

            second_term = - beta.sum()
            #second_term = Print('second_term')(second_term)

            return first_term + second_term
        #end if gconv + convmaxpool
        raise NotImplementedError()
    #end score matching



class GaussianConvolutionalVisLayer(Layer):
    def __init__(self,
            rows,
            cols,
            channels,
            init_beta = 1.,
            min_beta = 1.,
            init_mu = 0.,
            tie_beta = None,
            tie_mu = None):
        """
            Implements a visible layer that is conditionally gaussian with
            diagonal variance. The layer lives in a Conv2DSpace.

            rows, cols, channels: the shape of the space

            init_beta: the initial value of the precision parameter
            min_beta: clip beta so it is at least this big (default 1)

            init_mu: the initial value of the mean parameter

            tie_beta: None or a string specifying how to tie beta
                      'locations' = tie beta across locations, ie
                                    beta should be a vector with one
                                    elem per channel
            tie_mu: None or a string specifying how to tie mu
                    'locations' = tie mu across locations, ie
                                  mu should be a vector with one
                                  elem per channel

        """

        self.__dict__.update(locals())
        del self.self

        self.space = Conv2DSpace(shape = [rows,cols], nchannels = channels)
        self.input_space = self.space

        origin = self.space.get_origin()

        beta_origin = origin.copy()
        assert tie_beta in [ None, 'locations']
        if tie_beta == 'locations':
            beta_origin = beta_origin[0,0,:]
        self.beta = sharedX( beta_origin + init_beta,name = 'beta')
        assert self.beta.ndim == beta_origin.ndim

        mu_origin = origin.copy()
        assert tie_mu in [None, 'locations']
        if tie_mu == 'locations':
            mu_origin = mu_origin[0,0,:]
        self.mu = sharedX( mu_origin + init_mu, name = 'mu')
        assert self.mu.ndim == mu_origin.ndim

    def get_params(self):
        if self.mu is None:
            return set([self.beta])
        return set([self.beta, self.mu])

    def get_lr_scalers(self):
        rval = {}
        warn = False

        rows, cols = self.space.shape
        num_loc = float(rows * cols)

        assert self.tie_beta in [None, 'locations']
        if self.tie_beta == 'locations':
            warn = True
            rval[self.beta] = 1./num_loc

        assert self.tie_mu in [None, 'locations']
        if self.tie_mu == 'locations':
            warn = True
            rval[self.mu] = 1./num_loc

        if warn:
            warnings.warn("beta/mu lr_scalars hardcoded to 1/sharing")

        return rval

    def censor_updates(self, updates):
        if self.beta in updates:
            updates[self.beta] = T.clip(updates[self.beta],
                    self.min_beta,1e6)

    def init_inpainting_state(self, V, drop_mask, noise = False, return_unmasked = False):

        """for Vv, drop_mask_v in get_debug_values(V, drop_mask):
            assert Vv.ndim == 4
            assert drop_mask_v.ndim in [3,4]
            for i in xrange(drop_mask.ndim):
                if Vv.shape[i] != drop_mask_v.shape[i]:
                    print Vv.shape
                    print drop_mask_v.shape
                    assert False
        """

        if self.tie_mu == 'locations':
            unmasked = self.mu.dimshuffle('x', 'x', 'x', 0)
        else:
            assert self.tie_mu is None
            unmasked = self.mu.dimshuffle('x', 0, 1, 2)
        masked_mu = unmasked * drop_mask
        masked_mu = block_gradient(masked_mu)
        masked_mu.name = 'masked_mu'

        if noise:
            theano_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(42)
            unmasked = theano_rng.normal(avg = 0.,
                    std = 1., size = masked_mu.shape,
                    dtype = masked_mu.dtype)
            masked_mu = unmasked * drop_mask
            masked_mu.name = 'masked_noise'


        masked_V  = V  * (1-drop_mask)
        rval = masked_mu + masked_V
        rval.name = 'init_inpainting_state'

        if return_unmasked:
            return rval, unmasked
        return rval


    def expected_energy_term(self, state, average, state_below = None, average_below = None):
        assert state_below is None
        assert average_below is None
        self.space.validate(state)
        if average:
            raise NotImplementedError(str(type(self))+" doesn't support integrating out variational parameters yet.")
        else:
            rval =  0.5 * (self.beta * T.sqr(state - self.mu)).sum(axis=(1,2,3))
        assert rval.ndim == 1
        return rval


    def inpaint_update(self, state_above, layer_above, drop_mask = None, V = None,
                        return_unmasked = False):

        msg = layer_above.downward_message(state_above)
        mu = self.mu

        z = msg + mu
        z.name = 'inpainting_z_[unknown_iter]'

        if drop_mask is not None:
            rval = drop_mask * z + (1-drop_mask) * V
        else:
            rval = z


        rval.name = 'inpainted_V[unknown_iter]'

        if return_unmasked:
            return rval, z

        return rval

    def get_sampling_updates(self, state_below = None, state_above = None,
            layer_above = None,
            theano_rng = None):

        assert state_below is None
        msg = layer_above.downward_message(state_above)
        mu = self.mu

        z = msg + mu

        rval = theano_rng.normal(size = z.shape, avg = z, dtype = z.dtype,
                       std = 1. / T.sqrt(self.beta) )

        return rval

    def recons_cost(self, V, V_hat_unmasked, drop_mask = None):

        V_hat = V_hat_unmasked

        assert V.ndim == V_hat.ndim
        unmasked_cost = 0.5 * self.beta * T.sqr(V-V_hat) - 0.5*T.log(self.beta / (2*np.pi))
        assert unmasked_cost.ndim == V_hat.ndim

        if drop_mask is None:
            masked_cost = unmasked_cost
        else:
            masked_cost = drop_mask * unmasked_cost

        return masked_cost.mean()

    def upward_state(self, total_state):
        if total_state.ndim != 4:
            raise ValueError("total_state should have 4 dimensions, has "+str(total_state.ndim))
        assert total_state is not None
        V = total_state
        upward_state = V * self.beta
        return upward_state

    def make_state(self, num_examples, numpy_rng):

        rows, cols = self.space.shape
        channels = self.space.nchannels

        sample = numpy_rng.randn(num_examples, rows, cols, channels)

        sample *= 1./np.sqrt(self.beta.get_value())
        sample += self.mu.get_value()

        rval = sharedX(sample, name = 'v_sample_shared')

        return rval



class ConvMaxPool(HiddenLayer):
    def __init__(self,
             output_channels,
            kernel_rows,
            kernel_cols,
            pool_rows,
            pool_cols,
            irange,
            layer_name,
            scale_by_sharing = True,
            mirror_weights = False,
            init_bias = 0.,
            border_mode = 'valid',
            output_axes = ('b', 'c', 0, 1)):
        """

        mirror_weights:
            if true, initializes kernel i to be the negation of kernel i - 1
                I found this helps resolve a problem where the initial kernels often
                had a tendency to take grey inputs and turn them a different color,
                such as red

        """
        self.__dict__.update(locals())
        del self.self

        self.b = sharedX( np.zeros((output_channels,)) + init_bias, name = layer_name + '_b')
        assert border_mode in ['full','valid']

    def broadcasted_bias(self):

        assert self.b.ndim == 1

        shuffle = [ 'x' ] * 4
        shuffle[self.output_axes.index('c')] = 0

        return self.b.dimshuffle(*shuffle)

    def set_input_space(self, space):
        """ Note: this resets parameters!"""
        if not isinstance(space, Conv2DSpace):
            raise TypeError("ConvMaxPool can only act on a Conv2DSpace, but received " +
                    str(type(space))+" as input.")
        self.input_space = space
        self.input_rows, self.input_cols = space.shape
        self.input_channels = space.nchannels

        if self.border_mode == 'valid':
            self.h_rows = self.input_rows - self.kernel_rows + 1
            self.h_cols = self.input_cols - self.kernel_cols + 1
        else:
            assert self.border_mode == 'full'
            self.h_rows = self.input_rows + self.kernel_rows - 1
            self.h_cols = self.input_cols + self.kernel_cols - 1


        if not( self.h_rows % self.pool_rows == 0):
            raise ValueError("h_rows = %d, pool_rows = %d. Should be divisible but remainder is %d" %
                    (self.h_rows, self.pool_rows, self.h_rows % self.pool_rows))
        assert self.h_cols % self.pool_cols == 0

        self.h_space = Conv2DSpace(shape = (self.h_rows, self.h_cols), nchannels = self.output_channels,
                axes = self.output_axes)
        self.output_space = Conv2DSpace(shape = (self.h_rows / self.pool_rows,
                                                self.h_cols / self.pool_cols),
                                                nchannels = self.output_channels,
                axes = self.output_axes)

        print self.layer_name,': detector shape:',self.h_space.shape,'pool shape:',self.output_space.shape

        if tuple(self.output_axes) == ('b', 0, 1, 'c'):
            self.max_pool = max_pool_b01c
        elif tuple(self.output_axes) == ('b', 'c', 0, 1):
            self.max_pool = max_pool
        else:
            raise NotImplementedError()

        self.transformer = make_random_conv2D(self.irange, input_space = space,
                output_space = self.h_space, kernel_shape = (self.kernel_rows, self.kernel_cols),
                batch_size = self.dbm.batch_size, border_mode = self.border_mode)
        self.transformer._filters.name = self.layer_name + '_W'

        if self.mirror_weights:
            filters = self.transformer._filters
            v = filters.get_value()
            for i in xrange(1, v.shape[0], 2):
                v[i, :, :, :] = - v[i-1,:, :, :].copy()
            filters.set_value(v)

        W ,= self.transformer.get_params()
        assert W.name is not None

    def get_params(self):
        assert self.b.name is not None
        W ,= self.transformer.get_params()
        assert W.name is not None
        return self.transformer.get_params().union([self.b])

    def state_to_b01c(self, state):

        if tuple(self.output_axes) == ('b',0,1,'c'):
            return state
        return [ Conv2DSpace.convert(elem, self.output_axes, ('b', 0, 1, 'c'))
                for elem in state ]

    def get_l1_act_cost(self, state, target, coeff, eps):
        """

            target: if pools contain more than one element, should be a list with
                    two elements. the first element is for the pooling units and
                    the second for the detector units.

        """
        rval = 0.

        if self.pool_rows == 1 and self.pool_cols == 1:
            # If the pool size is 1 then pools = detectors
            # and we should not penalize pools and detectors separately
            assert len(state) == 2
            assert isinstance(target, float)
            assert isinstance(coeff, float)
            _, state = state
            state = [state]
            target = [target]
            coeff = [coeff]
            eps = [eps]
        else:
            assert all([len(elem) == 2 for elem in [state, target, coeff]])
            if target[1] < target[0]:
                warnings.warn("Do you really want to regularize the detector units to be sparser than the pooling units?")

        for s, t, c, e in safe_zip(state, target, coeff, eps):
            if c == 0.:
                continue
            # Average over everything but the channel index
            m = s.mean(axis= [ ax for ax in range(4) if self.output_axes[ax] != 'c' ])
            assert m.ndim == 1
            rval += T.maximum(abs(m-t)-e,0.).mean()*c

        return rval

    def get_lr_scalers(self):
        if self.scale_by_sharing:
            # scale each learning rate by 1 / # times param is reused
            h_rows, h_cols = self.h_space.shape
            num_h = float(h_rows * h_cols)
            return { self.transformer._filters : 1./num_h,
                     self.b : 1. / num_h  }
        else:
            return {}

    def upward_state(self, total_state):
        p,h = total_state
        return p

    def downward_state(self, total_state):
        p,h = total_state
        return h

    def get_monitoring_channels_from_state(self, state):

        P, H = state

        if tuple(self.output_axes) == ('b',0,1,'c'):
            p_max = P.max(axis=(0,1,2))
            p_min = P.min(axis=(0,1,2))
            p_mean = P.mean(axis=(0,1,2))
        else:
            assert tuple(self.output_axes) == ('b','c',0,1)
            p_max = P.max(axis=(0,2,3))
            p_min = P.min(axis=(0,2,3))
            p_mean = P.mean(axis=(0,2,3))
        p_range = p_max - p_min

        rval = {
                'p_max_max' : p_max.max(),
                'p_max_mean' : p_max.mean(),
                'p_max_min' : p_max.min(),
                'p_min_max' : p_min.max(),
                'p_min_mean' : p_min.mean(),
                'p_min_max' : p_min.max(),
                'p_range_max' : p_range.max(),
                'p_range_mean' : p_range.mean(),
                'p_range_min' : p_range.min(),
                'p_mean_max' : p_mean.max(),
                'p_mean_mean' : p_mean.mean(),
                'p_mean_min' : p_mean.min()
                }

        return rval



    def mf_update(self, state_below, state_above, layer_above = None, double_weights = False, iter_name = None):

        self.input_space.validate(state_below)

        if iter_name is None:
            iter_name = 'anon'

        if state_above is not None:
            assert layer_above is not None
            msg = layer_above.downward_message(state_above)
            msg.name = 'msg_from_'+layer_above.layer_name+'_to_'+self.layer_name+'['+iter_name+']'
        else:
            msg = None

        if not hasattr(state_below, 'ndim'):
            raise TypeError("state_below should be a TensorType, got " +
                    str(state_below) + " of type " + str(type(state_below)))
        if state_below.ndim != 4:
            raise ValueError("state_below should have ndim 4, has "+str(state_below.ndim))

        if double_weights:
            state_below = 2. * state_below
            state_below.name = self.layer_name + '_'+iter_name + '_2state'
        z = self.transformer.lmul(state_below) + self.broadcasted_bias()
        if self.layer_name is not None and iter_name is not None:
            z.name = self.layer_name + '_' + iter_name + '_z'
        p,h = self.max_pool(z, (self.pool_rows, self.pool_cols), msg)

        p.name = self.layer_name + '_p_' + iter_name
        h.name = self.layer_name + '_h_' + iter_name

        return p, h

    def get_sampling_updates(self, state_below = None, state_above = None,
            layer_above = None,
            theano_rng = None):

        if state_above is not None:
            msg = layer_above.downward_message(state_above)
        else:
            msg = None

        z = self.transformer.lmul(state_below) + self.broadcasted_bias()
        p, h, p_sample, h_sample = self.max_pool(z,
                (self.pool_rows, self.pool_cols), msg, theano_rng)

        return p_sample, h_sample

    def downward_message(self, downward_state):
        self.h_space.validate(downward_state)
        return self.transformer.lmul_T(downward_state)

    def set_batch_size(self, batch_size):
        self.transformer.set_batch_size(batch_size)

    def get_weights_topo(self):
        outp, inp, rows, cols = range(4)
        raw = self.transformer._filters.get_value()

        return np.transpose(raw,(outp,rows,cols,inp))


    def make_state(self, num_examples, numpy_rng):
        """ Returns a shared variable containing an actual state
           (not a mean field state) for this variable.
        """

        t1 = time.time()

        empty_input = self.h_space.get_origin_batch(self.dbm.batch_size)
        h_state = sharedX(empty_input)

        default_z = T.zeros_like(h_state) + self.broadcasted_bias()

        theano_rng = MRG_RandomStreams(numpy_rng.randint(2 ** 16))

        p_exp, h_exp, p_sample, h_sample = self.max_pool(
                z = default_z,
                pool_shape = (self.pool_rows, self.pool_cols),
                theano_rng = theano_rng)

        p_state = sharedX( self.output_space.get_origin_batch(
            self.dbm.batch_size))


        t2 = time.time()

        f = function([], updates = {
            p_state : p_sample,
            h_state : h_sample
            })

        t3 = time.time()

        f()

        t4 = time.time()

        print str(self)+'.make_state took',t4-t1
        print '\tcompose time:',t2-t1
        print '\tcompile time:',t3-t2
        print '\texecute time:',t4-t3

        p_state.name = 'p_sample_shared'
        h_state.name = 'h_sample_shared'

        return p_state, h_state

    def expected_energy_term(self, state, average, state_below, average_below):

        self.input_space.validate(state_below)

        downward_state = self.downward_state(state)
        self.h_space.validate(downward_state)

        # Energy function is linear so it doesn't matter if we're averaging or not
        # Specifically, our terms are -u^T W d - b^T d where u is the upward state of layer below
        # and d is the downward state of this layer

        bias_term = (downward_state * self.broadcasted_bias()).sum(axis=(1,2,3))
        weights_term = (self.transformer.lmul(state_below) * downward_state).sum(axis=(1,2,3))

        rval = -bias_term - weights_term

        assert rval.ndim == 1

        return rval


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

    def init_inpainting_state(self, Y, noise):
        if noise:
            theano_rng = MRG_RandomStreams(2012+10+30)
            return T.nnet.softmax(theano_rng.normal(avg=0., size=Y.shape, std=1., dtype='float32'))
        return T.nnet.softmax(self.b)



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
        return self.super_dbm.get_params().union(self.extra_layer.get_params())

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
        assert isinstance(model.hidden_layers[-1], Softmax)
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

    def get_gradients(self, model, X, Y, **kwargs):

        new_kwargs = { 'niter' : self.grad_niter }
        new_kwargs.update(kwargs)

        cost = self(model, X, Y, ** new_kwargs)

        params = list(model.get_params())
        grads = dict(safe_zip(params, T.grad(cost, params, disconnected_inputs = 'ignore')))

        return grads, {}

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

class CompositeLayer(HiddenLayer):
    """
        A Layer constructing by aligning several other Layer
        objects side by side
    """

    def __init__(self, layer_name, components, inputs_to_components = None):
        """
            components: A list of layers that are combined to form this layer
            inputs_to_components: None or dict mapping int to list of int
                Should be None unless the input space is a CompositeSpace
                If inputs_to_components[i] contains j, it means input i will
                be given as input to component j.
                If an input dodes not appear in the dictionary, it will be given
                to all components.

                This field allows one CompositeLayer to have another as input
                without forcing each component to connect to all members
                of the CompositeLayer below. For example, you might want to
                have both densely connected and convolutional units in all
                layers, but a convolutional unit is incapable of taking a
                non-topological input space.
        """

        self.layer_name = layer_name

        self.components = list(components)
        assert isinstance(components, list)
        for component in components:
            assert isinstance(component, SuperDBM_HidLayer)
        self.num_components = len(components)
        self.components = list(components)

        if inputs_to_components is None:
            self.inputs_to_components = None
        else:
            if not isinstance(inputs_to_components, dict):
                raise TypeError("CompositeLayer expected inputs_to_components to be a dict, got "+str(type(inputs_to_components)))
            self.inputs_to_components = {}
            for key in inputs_to_components:
                assert isinstance(key, int)
                assert key >= 0
                value = inputs_to_components[key]
                assert isinstance(value, list)
                assert all([isinstance(elem, int) for elem in value])
                assert min(value) >= 0
                assert max(value) < self.num_components
                self.inputs_to_components[key] = list(value)

    def set_input_space(self, space):

        self.input_space = space

        if not isinstance(space, CompositeSpace):
            assert self.inputs_to_components is None
            self.routing_needed = False
        else:
            if self.inputs_to_components is None:
                self.routing_needed = False
            else:
                self.routing_needed = True
                assert max(self.inputs_to_components) < space.num_components
                # Invert the dictionary
                self.components_to_inputs = {}
                for i in xrange(self.num_components):
                    inputs = []
                    for j in xrange(space.num_components):
                        if i in self.inputs_to_components[j]:
                            inputs.append(i)
                    if len(inputs) < space.num_components:
                        self.components_to_inputs[i] = inputs

        for i, component in enumerate(self.components):
            if self.routing_needed and i in self.components_to_inputs:
                cur_space = space.restrict(self.components_to_inputs[i])
            else:
                cur_space = space

            component.set_input_space(cur_space)

        self.output_space = CompositeSpace([ component.get_output_space() for component in self.components ])

    def set_batch_size(self, batch_size):
        for component in self.components:
            component.set_batch_size(batch_size)

    def set_dbm(self, dbm):
        for component in self.components:
            component.set_dbm(dbm)

    def mf_update(self, state_below, state_above, layer_above = None, double_weights = False, iter_name = None):

        rval = []

        for i, component in enumerate(self.components):
            if self.routing_needed and i in self.components_to_inputs:
                cur_state_below =self.input_space.restrict_batch(state_below, self.components_to_inputs[i])
            else:
                cur_state_below = state_below

            class RoutingLayer(object):
                def __init__(self, idx, layer):
                    self.__dict__.update(locals())
                    del self.self
                    self.layer_name = 'route_'+str(idx)+'_'+layer.layer_name

                def downward_message(self, state):
                    return self.layer.downward_message(state)[self.idx]

            if layer_above is not None:
                cur_layer_above = RoutingLayer(i, layer_above)
            else:
                cur_layer_above = None

            mf_update = component.mf_update(state_below = cur_state_below,
                                            state_above = state_above,
                                            layer_above = cur_layer_above,
                                            double_weights = double_weights,
                                            iter_name = iter_name)

            rval.append(mf_update)

        return tuple(rval)

    def upward_state(self, total_state):

        return tuple([component.upward_state(elem)
            for component, elem in
            safe_zip(self.components, total_state)])

    def downward_state(self, total_state):

        return tuple([component.downward_state(elem)
            for component, elem in
            safe_zip(self.components, total_state)])

    def downward_message(self, downward_state):

        if isinstance(self.input_space, CompositeSpace):
            num_input_components = self.input_space.num_components
        else:
            num_input_components = 1

        rval = [ None ] * num_input_components

        def add(x, y):
            if x is None:
                return y
            if y is None:
                return x
            return x + y

        for i, packed in enumerate(safe_zip(self.components, downward_state)):
            component, state = packed
            if self.routing_needed and i in self.components_to_inputs:
                input_idx = self.components_to_inputs[i]
            else:
                input_idx = range(num_input_components)

            partial_message = component.downward_message(state)

            if len(input_idx) == 1:
                partial_message = [ partial_message ]

            assert len(input_idx) == len(partial_message)

            for idx, msg in safe_zip(input_idx, partial_message):
                rval[idx] = add(rval[idx], msg)

        if len(rval) == 1:
            rval = rval[0]
        else:
            rval = tuple(rval)

        self.input_space.validate(rval)

        return rval

    def get_l1_act_cost(self, state, target, coeff, eps):
        return sum([ comp.get_l1_act_cost(s, t, c, e) \
            for comp, s, t, c, e in safe_zip(self.components, state, target, coeff, eps)])

    def get_params(self):
        return reduce(lambda x, y: x.union(y),
                [component.get_params() for component in self.components])

    def get_weights_topo(self):
        print 'Get topological weights for which layer?'
        for i, component in enumerate(self.components):
            print i,component.layer_name
        x = raw_input()
        return self.components[int(x)].get_weights_topo()

    def get_monitoring_channels_from_state(self, state):
        rval = {}

        for layer, s in safe_zip(self.components, state):
            d = layer.get_monitoring_channels_from_state(s)
            for key in d:
                rval[layer.layer_name+'_'+key] = d[key]

        return rval


class BinaryVisLayer(dbm.BinaryVector):

    def init_inpainting_state(self, V, drop_mask, noise = False, return_unmasked = False):

        assert drop_mask.ndim > 1

        unmasked = T.nnet.sigmoid(self.bias.dimshuffle('x',0))
        # this condition is needed later if unmasked is used as V_hat
        assert unmasked.ndim == 2
        # this condition is also needed later if unmasked is used as V_hat
        assert hasattr(unmasked.owner.op, 'scalar_op')
        masked_mean = unmasked * drop_mask
        masked_mean = block_gradient(masked_mean)
        masked_mean.name = 'masked_mean'

        if noise:
            theano_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(42)
            # we want a set of random mean field parameters, not binary samples
            unmasked = T.nnet.sigmoid(theano_rng.normal(avg = 0.,
                    std = 1., size = masked_mean.shape,
                    dtype = masked_mean.dtype))
            masked_mean = unmasked * drop_mask
            masked_mean.name = 'masked_noise'


        masked_V  = V  * (1-drop_mask)
        rval = masked_mean + masked_V
        rval.name = 'init_inpainting_state'

        if return_unmasked:
            assert unmasked.ndim > 1
            return rval, unmasked

        return rval


    def inpaint_update(self, state_above, layer_above, drop_mask = None, V = None, return_unmasked = False):

        msg = layer_above.downward_message(state_above)
        mu = self.bias

        z = msg + mu
        z.name = 'inpainting_z_[unknown_iter]'

        unmasked = T.nnet.sigmoid(z)

        if drop_mask is not None:
            rval = drop_mask * unmasked + (1-drop_mask) * V
        else:
            rval = unmasked

        rval.name = 'inpainted_V[unknown_iter]'

        if return_unmasked:
            owner = unmasked.owner
            assert owner is not None
            op = owner.op
            assert hasattr(op, 'scalar_op')
            assert isinstance(op.scalar_op, T.nnet.sigm.ScalarSigmoid)
            return rval, unmasked

        return rval


    def recons_cost(self, V, V_hat_unmasked, drop_mask = None):

        V_hat = V_hat_unmasked

        assert hasattr(V_hat, 'owner')
        owner = V_hat.owner
        assert owner is not None
        op = owner.op
        block_grad = False
        if isinstance(op, _ElemwiseNoGradient):
            assert isinstance(op.scalar_op, theano.scalar.Identity)
            block_grad = True
            real, = owner.inputs
            owner = real.owner
            op = owner.op

        if not hasattr(op, 'scalar_op'):
            raise ValueError("Expected V_hat_unmasked to be generated by an Elemwise op, got "+str(op)+" of type "+str(type(op)))
        assert isinstance(op.scalar_op, T.nnet.sigm.ScalarSigmoid)
        z ,= owner.inputs
        if block_grad:
            z = block_gradient(z)

        if V.ndim != V_hat.ndim:
            raise ValueError("V and V_hat_unmasked should have same ndim, but are %d and %d." % (V.ndim, V_hat.ndim))
        unmasked_cost = V * T.nnet.softplus(-z) + (1 - V) * T.nnet.softplus(z)
        assert unmasked_cost.ndim == V_hat.ndim

        if drop_mask is None:
            masked_cost = unmasked_cost
        else:
            masked_cost = drop_mask * unmasked_cost

        return masked_cost.mean()


class DBM_PCD(Cost):
    """
    An intractable cost representing the variational upper bound
    on the negative log likelihood of a DBM.
    The gradient of this bound is computed using a persistent
    markov chain.
    """

    def __init__(self, num_chains, num_gibbs_steps, supervised = False):
        self.__dict__.update(locals())
        del self.self
        self.theano_rng = MRG_RandomStreams(2012 + 10 + 14)
        assert supervised in [True, False]

    def __call__(self, model, X, Y=None):
        """
        The partition function makes this intractable.
        """

        if self.supervised:
            assert Y is not None

        return None

    def get_monitoring_channels(self, model, X, Y = None):
        rval = {}

        history = model.mf(X, return_history = True)
        q = history[-1]

        if len(history) > 1:
            prev_q = history[-2]

            flat_q = flatten(q)
            flat_prev_q = flatten(prev_q)

            mx = None
            for new, old in safe_zip(flat_q, flat_prev_q):
                cur_mx = abs(new - old).max()
                if mx is None:
                    mx = cur_mx
                else:
                    mx = T.maximum(mx, cur_mx)

            rval['max_var_param_diff'] = mx

        if self.supervised:
            assert Y is not None
            Y_hat = q[-1]
            true = T.argmax(Y,axis=1)
            pred = T.argmax(Y_hat, axis=1)

            #true = Print('true')(true)
            #pred = Print('pred')(pred)

            wrong = T.neq(true, pred)
            err = T.cast(wrong.mean(), X.dtype)
            rval['err'] = err

        return rval



    def get_gradients(self, model, X, Y=None):
        """
        PCD approximation to the gradient of the bound.
        Keep in mind this is a cost, so we are upper bounding
        the negative log likelihood.
        """

        if self.supervised:
            assert Y is not None
            # note: if the Y layer changes to something without linear energy,
            # we'll need to make the expected energy clamp Y in the positive phase
            assert isinstance(model.hidden_layers[-1], Softmax)



        q = model.mf(X, Y)


        """
            Use the non-negativity of the KL divergence to construct a lower bound
            on the log likelihood. We can drop all terms that are constant with
            repsect to the model parameters:

            log P(v) = L(v, q) + KL(q || P(h|v))
            L(v, q) = log P(v) - KL(q || P(h|v))
            L(v, q) = log P(v) - sum_h q(h) log q(h) + q(h) log P(h | v)
            L(v, q) = log P(v) + sum_h q(h) log P(h | v) + const
            L(v, q) = log P(v) + sum_h q(h) log P(h, v) - sum_h q(h) log P(v) + const
            L(v, q) = sum_h q(h) log P(h, v) + const
            L(v, q) = sum_h q(h) -E(h, v) - log Z + const

            so the cost we want to minimize is
            expected_energy + log Z + const


            Note: for the RBM, this bound is exact, since the KL divergence goes to 0.
        """

        variational_params = flatten(q)

        # The gradients of the expected energy under q are easy, we can just do that in theano
        expected_energy_q = model.expected_energy(X, q).mean()
        params = list(model.get_params())
        gradients = dict(safe_zip(params, T.grad(expected_energy_q, params,
            consider_constant = variational_params,
            disconnected_inputs = 'ignore')))

        """
        d/d theta log Z = (d/d theta Z) / Z
                        = (d/d theta sum_h sum_v exp(-E(v,h)) ) / Z
                        = (sum_h sum_v - exp(-E(v,h)) d/d theta E(v,h) ) / Z
                        = - sum_h sum_v P(v,h)  d/d theta E(v,h)
        """

        layer_to_chains = model.make_layer_to_state(self.num_chains)

        def recurse_check(l):
            if isinstance(l, (list, tuple)):
                for elem in l:
                    recurse_check(elem)
            else:
                assert l.get_value().shape[0] == self.num_chains

        recurse_check(layer_to_chains.values())

        model.layer_to_chains = layer_to_chains

        orig_V = layer_to_chains[model.visible_layer] # rm
        orig_Y = layer_to_chains[model.hidden_layers[-1]] # rm

        # Note that we replace layer_to_chains with a dict mapping to the new
        # state of the chains
        updates, layer_to_chains = model.get_sampling_updates(layer_to_chains,
                self.theano_rng, num_steps=self.num_gibbs_steps,
                return_layer_to_updated = True)

        assert self.num_gibbs_steps >= len(model.hidden_layers) # rm
        assert orig_Y in theano.gof.graph.ancestors([layer_to_chains[model.visible_layer]]) # rm
        assert updates[orig_V] is layer_to_chains[model.visible_layer] #rm

        warnings.warn("""TODO: reduce variance of negative phase by integrating out
                the even-numbered layers. The Rao-Blackwellize method can do this
                for you when expected gradient = gradient of expectation, but doing
                this in general is trickier.""")
        #layer_to_chains = model.rao_blackwellize(layer_to_chains)

        expected_energy_p = model.energy(layer_to_chains[model.visible_layer],
                [layer_to_chains[layer] for layer in model.hidden_layers]).mean()

        samples = flatten(layer_to_chains.values())
        for i, sample in enumerate(samples):
            if sample.name is None:
                sample.name = 'sample_'+str(i)

        neg_phase_grads = dict(safe_zip(params, T.grad(-expected_energy_p, params, consider_constant
            = samples, disconnected_inputs='ignore')))


        for param in list(gradients.keys()):
            #print param.name,': '
            #print theano.printing.min_informative_str(neg_phase_grads[param])
            gradients[param] =  neg_phase_grads[param]  + gradients[param]

        return gradients, updates


class MF_L1_ActCost(Cost):
    """
    L1 activation cost on the mean field parameters.

    Adds a cost of:

    coeff * max( abs(mean_activation - target) - eps, 0)

    averaged over units

    for each layer.

    """

    def __init__(self, targets, coeffs, eps):
        """
        targets: a list, one element per layer, specifying the activation
                each layer should be encouraged to have
                    each element may also be a list depending on the
                    structure of the layer.
                See each layer's get_l1_act_cost for a specification of
                    what the state should be.
        coeffs: a list, one element per layer, specifying the coefficient
                to put on the L1 activation cost for each layer
        """
        self.__dict__.update(locals())
        del self.self

    def __call__(self, model, X, Y = None, ** kwargs):

        H_hat = model.mf(X)

        assert len(H_hat) > 0

        layer_costs = []
        for layer, mf_state, targets, coeffs, eps in \
            safe_zip(model.hidden_layers, H_hat, self.targets, self.coeffs, self.eps):
            cost = None
            try:
                cost = layer.get_l1_act_cost(mf_state, targets, coeffs, eps)
            except NotImplementedError:
                assert isinstance(coeffs, float) and coeffs == 0.
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

class DBM_WeightDecay(Cost):
    """
    coeff * sum(sqr(weights))

    for each set of weights.

    """

    def __init__(self, coeffs):
        """
        coeffs: a list, one element per layer, specifying the coefficient
                to put on the L1 activation cost for each layer.
                Each element may in turn be a list, ie, for CompositeLayers.
        """
        self.__dict__.update(locals())
        del self.self

    def __call__(self, model, X, Y = None, ** kwargs):

        layer_costs = [ layer.get_weight_decay(coeff)
            for layer, coeff in safe_izip(model.hidden_layers, self.coeffs) ]

        assert T.scalar() != 0. # make sure theano semantics do what I want
        layer_costs = [ cost for cost in layer_costs if cost != 0.]

        if len(layer_costs) == 0:
            return T.as_tensor_variable(0.)
        else:
            total_cost = reduce(lambda x, y: x + y, layer_costs)
        total_cost.name = 'DBM_WeightDecay'

        assert total_cost.ndim == 0

        return total_cost

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
            train_rnn_y = False, gibbs_features = False):
        assert initially_freeze_lower in [True, False, 0, 1]
        assert decapitate in [True, False, 0, 1]
        assert train_rnn_y in [True, False, 0, 1]
        self.__dict__.update(locals())

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
        assert isinstance(l1, DenseMaxPool)
        assert isinstance(l2, DenseMaxPool)
        if self.orig_sup:
            assert isinstance(c, Softmax)

        self._params = []

        # Layer 1
        self.vishid = sharedX(l1.get_weights())
        self._params.append(self.vishid)
        self.hidbias = sharedX(l1.get_biases())
        self._params.append(self.hidbias)

        # Layer 2
        self.hidpen = sharedX(l2.get_weights())
        self._params.append(self.hidpen)
        self.penhid = sharedX(l2.get_weights().T)
        self._params.append(self.penhid)
        penbias = l2.get_biases()
        if decapitate:
            Wc = c.get_weights()
            penbias += np.dot(Wc,
                    np.ones((c.n_classes,), dtype = penbias.dtype) * decapitated_value / c.n_classes)
            l2.set_biases(penbias)
        self.penbias = sharedX(l2.get_biases())
        self._params.append(self.penbias)

        # Class layer
        if decapitate:
            self.c = c
            del super_dbm.hidden_layers[-1]
        else:
            self.c = Softmax(n_classes = 10, irange = 0., layer_name = 'final_output')
            self.c.dbm = l1.dbm
            self.c.set_input_space(l2.get_output_space())
            if self.orig_sup:
                self.c.set_weights(c.get_weights())
                self.c.set_biases(c.get_biases())
        self._params.extend(self.c.get_params())

        if train_rnn_y:
            assert not decapitate
            self._params.extend(c.get_params())

        if final_irange is not None:
            if self.orig_sup:
                del c # Make sure we don't modify the feature-generating RNN
            W = self.c.dbm.rng.uniform(-final_irange, final_irange,
                    (self.c.input_space.get_total_dimension(),
                        self.c.n_classes))
            self.c.set_weights(W.astype('float32'))
            self.c.set_biases(np.zeros((self.c.n_classes)).astype('float32'))

        self.hidden_layers = [ self.c]

        if initially_freeze_lower:
            lr_scalers = {}
            gate = sharedX(0.)
            for param in self._params:
                if param not in self.c.get_params():
                    lr_scalers[param] = gate
            self.lr_scalers = lr_scalers
        else:
            self.lr_scalers = {}

    def get_lr_scalers(self):
        return self.lr_scalers

    def mf(self, V, return_history = False, ** kwargs):
        assert not return_history
        if not hasattr(self, 'gibbs_features'):
            self.gibbs_features = False
        if not self.gibbs_features:
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
        _, H2 = H2
        below = T.dot(V, self.vishid)
        above = T.dot(H2, self.penhid)
        H1 = T.nnet.sigmoid(below + above + self.hidbias)
        H2 = T.nnet.sigmoid(T.dot(H1, self.hidpen) + self.penbias)
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

class ActivateLower(TrainingCallback):
    def __call__(self, model, dataset, algorithm):
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

        H1 = T.nnet.sigmoid(T.dot(V, 2. * self.vishid[0] + self.biashid[0]))
        H2 = T.nnet.sigmoid(T.dot(H1, 2 * self.hidpen[0] + self.biaspen[0]))
        Y = T.nnet.softmax(T.dot(H2, self.penlab[0] + self.biaslab[0]))

        for i in xrange(1, self.niter):
            H1 = T.nnet.sigmoid(T.dot(V, self.vishid[i])+T.dot(H2, self.penhid[i])+self.biashid[i])
            Y = T.nnet.softmax(T.dot(H2, self.penlab[i] + self.biaslab[i]))
            H2 = T.nnet.sigmoid(T.dot(H1, self.hidpen[i]) + T.dot(Y, self.labpen[i]) + self.biaspen[0])

        return [H1, H2, Y]


