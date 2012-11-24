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

    def get_weight_decay(self, coeffs):
        W , = self.transformer.get_params()
        return coeffs * T.sqr(W).sum()



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

    def sample(self, state_below = None, state_above = None,
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


    def init_mf_state(self):
        default_z = self.broadcasted_bias()
        shape = {
                'b': self.dbm.batch_size,
                0: self.h_space.shape[0],
                1: self.h_space.shape[1],
                'c': self.h_space.nchannels
                }
        # work around theano bug with broadcasted stuff
        default_z += T.alloc(*([0.]+[shape[elem] for elem in self.h_space.axes])).astype(default_z.dtype)
        assert default_z.ndim == 4

        p, h = self.max_pool(
                z = default_z,
                pool_shape = (self.pool_rows, self.pool_cols))

        return p, h

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
            assert isinstance(component, dbm.HiddenLayer)
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

    def init_mf_state(self):
        return tuple([component.init_mf_state() for component in self.components])


    def get_weight_decay(self, coeffs):
        return sum([component.get_weight_decay(coeff) for component, coeff
            in safe_zip(self.components, coeffs)])

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

    def get_range_rewards(self, state, coeffs):
        return sum([comp.get_range_rewards(s, c)
            for comp, s, c in safe_zip(self.components, state, coeffs)])

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



