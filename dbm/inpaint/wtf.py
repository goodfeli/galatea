from pylearn2.models.model import Model
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
from theano.printing import Print
from theano.printing import min_informative_str
from pylearn2.utils import block_gradient
import warnings
from theano import function
from theano.sandbox.rng_mrg import MRG_RandomStreams
import time
from pylearn2.costs.cost import Cost
from pylearn2.utils import safe_zip
from pylearn2.utils import safe_izip
from pylearn2.utils import _ElemwiseNoGradient
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

    def setup_inference_procedure(self):
        if not hasattr(self, 'inference_procedure') or \
                self.inference_procedure is None:
            self.inference_procedure = SuperWeightDoubling()
            self.inference_procedure.set_dbm(self)


    def mf(self, *args, **kwargs):
        self.setup_inference_procedure()
        return self.inference_procedure.mf(*args, **kwargs)

    def do_inpainting(self, *args, **kwargs):
        self.setup_inference_procedure()
        return self.inference_procedure.do_inpainting(*args, **kwargs)



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



class GaussianConvolutionalVisLayer(VisibleLayer):
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
            return [self.beta]
        return [self.beta, self.mu]

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

    def sample(self, state_below = None, state_above = None,
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
            layer_name,
            center = False,
            irange = None,
            sparse_init = None,
            scale_by_sharing = True,
            init_bias = 0.,
            border_mode = 'valid',
            output_axes = ('b', 'c', 0, 1)):
        """


        """
        self.__dict__.update(locals())
        del self.self

        assert (irange is None) != (sparse_init is None)

        self.b = sharedX( np.zeros((output_channels,)) + init_bias, name = layer_name + '_b')
        assert border_mode in ['full','valid']

    def broadcasted_bias(self):

        assert self.b.ndim == 1

        shuffle = [ 'x' ] * 4
        shuffle[self.output_axes.index('c')] = 0

        return self.b.dimshuffle(*shuffle)


    def get_total_state_space(self):
        return CompositeSpace((self.h_space, self.output_space))

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

        if self.irange is not None:
            self.transformer = make_random_conv2D(self.irange, input_space = space,
                    output_space = self.h_space, kernel_shape = (self.kernel_rows, self.kernel_cols),
                    batch_size = self.dbm.batch_size, border_mode = self.border_mode, rng = self.dbm.rng)
        else:
            self.transformer = make_sparse_random_conv2D(self.sparse_init, input_space = space,
                    output_space = self.h_space, kernel_shape = (self.kernel_rows, self.kernel_cols),
                    batch_size = self.dbm.batch_size, border_mode = self.border_mode, rng = self.dbm.rng)
        self.transformer._filters.name = self.layer_name + '_W'


        W ,= self.transformer.get_params()
        assert W.name is not None

        if self.center:
            p_ofs, h_ofs = self.init_mf_state()
            self.p_offset = sharedX(self.output_space.get_origin(), 'p_offset')
            self.h_offset = sharedX(self.h_space.get_origin(), 'h_offset')
            f = function([], updates={self.p_offset: p_ofs[0,:,:,:], self.h_offset: h_ofs[0,:,:,:]})
            f()


    def get_params(self):
        assert self.b.name is not None
        W ,= self.transformer.get_params()
        assert W.name is not None

        return [ W, self.b]

    def state_to_b01c(self, state):

        if tuple(self.output_axes) == ('b',0,1,'c'):
            return state
        return [ Conv2DSpace.convert(elem, self.output_axes, ('b', 0, 1, 'c'))
                for elem in state ]

    def get_range_rewards(self, state, coeffs):
        """
        TODO: WRITEME
        """
        rval = 0.

        if self.pool_rows == 1 and self.pool_cols == 1:
            # If the pool size is 1 then pools = detectors
            # and we should not penalize pools and detectors separately
            assert len(state) == 2
            assert isinstance(coeffs, float)
            _, state = state
            state = [state]
            coeffs = [coeffs]
        else:
            assert all([len(elem) == 2 for elem in [state, coeffs]])

        for s, c in safe_zip(state, coeffs):
            if c == 0.:
                continue
            # Range over everything but the channel index
            # theano can only take gradient through max if the max is over 1 axis or all axes
            # so I manually unroll the max for the case I use here
            assert self.h_space.axes == ('b', 'c', 0, 1)
            assert self.output_space.axes == ('b', 'c', 0, 1)
            mx = s.max(axis=3).max(axis=2).max(axis=0)
            assert hasattr(mx.owner.op, 'grad')
            mn = s.min(axis=3).max(axis=2).max(axis=0)
            assert hasattr(mn.owner.op, 'grad')
            assert mx.ndim == 1
            assert mn.ndim == 1
            r = mx - mn
            rval += (1. - r).mean() * c

        return rval

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
            if eps is None:
                eps = 0.
            eps = [eps]
        else:
            if eps is None:
                eps = [0., 0.]
            assert all([len(elem) == 2 for elem in [state, target, coeff]])
            if target[1] < target[0] and coeff[1] != 0.:
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

        if self.center:
            p -= self.p_offset
            h -= self.h_offset

        return p

    def downward_state(self, total_state):
        p,h = total_state

        if self.center:
            p -= self.p_offset
            h -= self.h_offset

        return h

