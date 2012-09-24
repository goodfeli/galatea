from pylearn2.models.model import Model
import theano
from pylearn2.space import Conv2DSpace
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX
from pylearn2.linear.conv2d import make_random_conv2D
import theano.tensor as T
import numpy as np
from galatea.dbm.inpaint.probabilistic_max_pooling import max_pool
from theano.gof.op import get_debug_values
from theano.printing import Print
from galatea.theano_upgrades import block_gradient
import warnings
from theano import function
from theano.sandbox.rng_mrg import MRG_RandomStreams
import time

warnings.warn('super_dbm changing the recursion limit')
import sys
sys.setrecursionlimit(50000)

class SuperDBM(Model):

    def __init__(self,
            batch_size,
            visible_layer,
            hidden_layers,
            niter):
        self.__dict__.update(locals())
        del self.self
        assert len(hidden_layers) >= 1
        for layer in hidden_layers:
            layer.dbm = self
        self._update_layer_input_spaces()
        self.force_batch_size = batch_size

    def _update_layer_input_spaces(self):
        """
            Tells each layer what its input space should be.
            Note: this usually resets the layer's parameters!
        """
        visible_layer = self.visible_layer
        hidden_layers = self.hidden_layers
        self.hidden_layers[0].set_input_space(visible_layer.space)
        for i in xrange(1,len(hidden_layers)):
            hidden_layers[i].set_input_space(hidden_layers[i-1].get_output_space())

    def get_params(self):

        for param in self.visible_layer.get_params():
            assert param.name is not None
        rval = self.visible_layer.get_params()
        for layer in self.hidden_layers:
            for param in layer.get_params():
                if param.name is None:
                    print type(layer)
                    assert False
            rval = rval.union(layer.get_params())
        return rval

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

        for layer in self.hidden_layers:
            layer.set_batch_size(batch_size)

    def censor_updates(self, updates):
        self.visible_layer.censor_updates(updates)
        for layer in self.hidden_layers:
            layer.censor_updates(updates)

    def get_input_space(self):
        return self.visible_layer.space

    def get_lr_scalers(self):
        rval = {}

        params = self.get_params()

        for layer in self.hidden_layers + [ self.visible_layer ]:
            contrib = layer.get_lr_scalers()

            # No two layers can contend to scale a parameter
            assert not any([key in rval for key in contrib])
            # Don't try to scale anything that's not a parameter
            assert all([key in params for key in contrib])

            rval.update(contrib)

        return rval

    def get_weights(self):
        if len(self.hidden_layers) == 1:
            return self.hidden_layers[0].get_weights()
        else:
            raise NotImplementedError()

    def get_weights_topo(self):
        return self.hidden_layers[0].get_weights_topo()

    def do_inpainting(self, V, drop_mask, return_history = False, noise = False):

        orig_V = V
        orig_drop_mask = drop_mask

        history = []

        V_hat = self.visible_layer.init_inpainting_state(V,drop_mask,noise)

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
                #layer_above = None,
                state_below = self.hidden_layers[-1].upward_state(H_hat[-1])))
        else:
            H_hat.append( self.hidden_layers[-1].mf_update(
                state_above = None,
                state_below = self.visible_layer.upward_state(V_hat)))

        def update_history():
            history.append( { 'V_hat' : V_hat, 'H_hat' : H_hat } )

        update_history()

        for i in xrange(self.niter-1):
            for j in xrange(0,len(H_hat),2):
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

            V_hat = self.visible_layer.inpaint_update(
                    state_above = self.hidden_layers[0].downward_state(H_hat[0]),
                    layer_above = self.hidden_layers[0],
                    V = V,
                    drop_mask = drop_mask)
            V_hat.name = 'V_hat[%d](V_hat = %s)' % (i, V_hat.name)

            for j in xrange(1,len(H_hat),2):
                state_below = self.hidden_layers[j-1].upward_state(H_hat[j-1])
                if j == len(H_hat) - 1:
                    state_above = None
                else:
                    state_above = self.hidden_layers[j+1].downward_state(H_hat[j+1])
                H_hat[j] = self.hidden_layers[j].mf_update(
                        state_below = state_below,
                        state_above = state_above)
                #end ifelse
            #end for j
            update_history()
        #end for i

        # debugging, make sure V didn't get changed in this function
        assert V is orig_V
        assert drop_mask is orig_drop_mask

        if return_history:
            return history
        else:
            return V_hat

    def mf(self, V, return_history = False):

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

        history = [ H_hat ]

        #we only need recurrent inference if there are multiple layers
        if len(H_hat) > 1:
            for i in xrange(self.niter-1):
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
                #end for j
                history.append(H_hat)
            #end for i

        if return_history:
            return history
        else:
            return H_hat

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
                    "there is no test that the code matches the lyx file.")

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




    def make_layer_to_state(self, num_examples):

        """ Makes and returns a dictionary mapping layers to states.
            By states, we mean here a real assignment, not a mean field state.
            For example, for a layer containing binary random variables, the
            state will be a shared variable containing values in {0,1}, not
            [0,1].
            The visible layer will be included.
            Uses a dictionary so it is easy to unambiguously index a layer
            without needing to remember rules like vis layer = 0, hiddens start
            at 1, etc.
        """

        # Make a list of all layers
        layers = [self.visible_layer] + self.hidden_layers

        rng = np.random.RandomState([2012,9,11])

        states = [ layer.make_state(num_examples, rng) for layer in layers ]

        rval = dict(zip(layers, states))

        return rval

    def get_sampling_updates(self, layer_to_state, theano_rng,
            layer_to_clamp = None):
        """
            layer_to_state: a dictionary mapping the SuperDBM_Layer instances
                            contained in self to shared variables representing
                            batches of samples of them.
                            (you can allocate one by calling
                            self.make_layer_to_state)
            theano_rng: a MRG_RandomStreams object
            layer_to_clamp: (optional) a dictionary mapping layers to bools
                            if a layer is not in the dictionary, defaults to False
                            True indicates that this layer should be clamped, so
                            we are sampling from a conditional distribution rather
                            than the joint
            returns a dictionary mapping each shared variable to an expression
                     to update it. Repeatedly applying these updates does MCMC
                     sampling.

            Note: this does Gibbs sampling, starting with the visible layer, and
            then working upward. If you initialize the visible sample with data,
            it will be discarded with no influence, since the visible layer is
            the first layer to be sampled
            sampled. To start Gibbs sampling from data you must do at least one
            sampling step explicitly clamping the visible units.
        """

        assert len(self.hidden_layers) > 0 # I guess we could make a model with
                                           # no latent layers if we really want

        if layer_to_clamp is None:
            layer_to_clamp = {}

        for key in layer_to_clamp:
            assert key is self.visible_layer or key in self.hidden_layers

        for layer in [ self.visible_layer ] + self.hidden_layers:
            if layer not in layer_to_clamp:
                layer_to_clamp[layer] = False

        rval = {}

        #Sample the visible layer
        vis_state = layer_to_state[self.visible_layer]
        if layer_to_clamp[self.visible_layer]:
            vis_sample = vis_state
        else:
            first_hid = self.hidden_layers[0]
            state_above = layer_to_state[first_hid]
            state_above = first_hid.downward_state(state_above)

            vis_sample = self.visible_layer.get_sampling_updates(
                    state_above = state_above,
                    layer_above = first_hid,
                    theano_rng = theano_rng)


        if isinstance(vis_state, (list, tuple)):
            for state, sample in zip(vis_state, vis_sample):
                rval[state] = sample
        else:
            rval[vis_state] = vis_sample

        for i in xrange(len(self.hidden_layers)):
            # Iteration i does the Gibbs step for hidden_layers[i]

            # Get the sampled state of the layer below so we can condition
            # on it in our Gibbs update
            if i == 0:
                layer_below = self.visible_layer
            else:
                layer_below = self.hidden_layers[i-1]

            state_below = layer_to_state[layer_below]
            # We want to sample from each conditional distribution
            # ***sequentially*** so we must use the updated version
            # of the state for the layers whose updates we have
            # calculcated already. If we used the raw value from
            # layer_to_state
            # then we would sample from each conditional
            # ***simultaneously*** which does not implement MCMC
            # sampling.
            if isinstance(state_below, (list,tuple)):
                state_below = tuple(
                        [rval[old_state] for old_state in state_below])
            else:
                state_below = rval[state_below]
                assert state_below is not None

            state_below = layer_below.upward_state(state_below)

            # Get the sampled state of the layer above so we can condition
            # on it in our Gibbs step
            if i + 1 < len(self.hidden_layers):
                layer_above = self.hidden_layers[i + 1]
                state_above = layer_to_state[layer_above]
                state_above = layer_above.downward_state(state_above)
            else:
                state_above = None
                layer_above = None

            # Compute the Gibbs sampling update
            # Sample the state of this layer conditioned
            # on its Markov blanket (the layer above and
            # layer below)
            this_layer = self.hidden_layers[i]
            this_sample = this_layer.get_sampling_updates(
                    state_below = state_below,
                    state_above = state_above,
                    layer_above = layer_above,
                    theano_rng = theano_rng)

            # Store the update in the dictionary, accounting for
            # composite states
            this_state = layer_to_state[this_layer]
            if layer_to_clamp[this_layer]:
                this_sample = this_state
            if isinstance(this_state, (list, tuple)):
                for state, sample in zip(this_state, this_sample):
                    assert hasattr(state,'get_value')
                    rval[state] = sample
            else:
                assert hasattr(this_state,'get_value')
                rval[this_state] = this_sample

        # Check that we updated all the samples
        states = set()
        for layer in layer_to_state:
            state_s = layer_to_state[layer]
            if isinstance(state_s, (list,tuple)):
                for state in state_s:
                    assert state in rval
                    states.add(state)
            else:
                assert state_s in rval
                states.add(state_s)
        # Check that we're not trying to update anything else
        for state in rval:
            assert hasattr(state,'get_value')
            if state not in states:
                print 'oops, you seem to be trying to update',state
                print 'but this does not seem to be a sampled state'
                assert False
        # Check that clamping worked
        # (We want rval to be the identity mapping here, so that the update
        # computations above can always use rval to refer to the state of
        # variables that have already been visited by the bottom-to-top
        # traversal)
        for layer in layer_to_clamp:
            if layer_to_clamp[layer]:
                state = layer_to_state[layer]
                if isinstance(state,(list,tuple)):
                    for elem in state:
                        assert rval[elem] is elem
                else:
                    assert rval[state] is state
        # Now that we know that clamping was respected, we actually want
        # to strip out the identity mapping, and just not have updates for
        # the clamped variables. This is so theano.function doesn't waste
        # time computing no-ops (not sure if theano would optimize these
        # out or not)
        for layer in layer_to_clamp:
            if layer_to_clamp[layer]:
                state = layer_to_state[layer]
                if isinstance(state,(list,tuple)):
                    for elem in state:
                        del rval[elem]
                else:
                    del rval[state]

        return rval


class SuperDBM_Layer(Model):

    def upward_state(self, total_state):
        """
            Takes total_state and turns it into the state that layer_above should
            see when computing P( layer_above | this_layer).

            So far this has two uses:
                If this layer consists of a detector sub-layer h that is pooled
                into a pooling layer p, then total_state = (p,h) but
                layer_above should only see p.

                If the conditional P( layer_above | this_layer) depends on
                parameters of this_layer, sometimes you can play games with
                the state to avoid needing the layers to communicate. So far
                the only instance of this usage is when the visible layer
                is N( Wh, beta). This makes the hidden layer be
                sigmoid( v beta W + b). Rather than having the hidden layer
                explicitly know about beta, we can just pass v beta as
                the upward state.

            Note: this method should work both for computing sampling updates
            and for computing mean field updates. So far I haven't encountered
            a case where it needs to do different things for those two
            contexts.
        """
        return total_state

    def make_state(self, num_examples, numpy_rng):
        """ Returns a shared variable containing an actual state
           (not a mean field state) for this variable.
        """

        raise NotImplementedError("%s doesn't implement make_state" %
                type(self))

    def get_sampling_updates(self, state_below = None, state_above = None,
            layer_above = None,
            theano_rng = None):
        """

            state_below is layer_below.upward_state(full_state_below)
            where full_state_below is the same kind of object as you get
            out of layer_below.make_state

            state_above is layer_above.downward_state(full_state_above)

            theano_rng is an MRG_RandomStreams instance

            Returns an expression for samples of this layer's state,
            conditioned on the layers above and below
            Should be valid as an update to the shared variable returned
            by self.make_state

            Note: this can return multiple expressions if this layer's
            total state consists of more than one shared variable
        """

        raise NotImplementedError("%s doesn't implement get_sampling_updates" %
                type(self))

class SuperDBM_HidLayer(SuperDBM_Layer):

    def downward_state(self, total_state):
        return total_state

class GaussianConvolutionalVisLayer(SuperDBM_Layer):
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

    def init_inpainting_state(self, V, drop_mask, noise = False):

        """for Vv, drop_mask_v in get_debug_values(V, drop_mask):
            assert Vv.ndim == 4
            assert drop_mask_v.ndim in [3,4]
            for i in xrange(drop_mask.ndim):
                if Vv.shape[i] != drop_mask_v.shape[i]:
                    print Vv.shape
                    print drop_mask_v.shape
                    assert False
        """

        masked_mu = self.mu * drop_mask
        masked_mu = block_gradient(masked_mu)
        masked_mu.name = 'masked_mu'

        if noise:
            theano_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(42)
            masked_mu = theano_rng.normal(avg = 0.,
                    std = 1., size = masked_mu.shape,
                    dtype = masked_mu.dtype) * drop_mask
            masked_mu.name = 'masked_noise'

        masked_V  = V  * (1-drop_mask)
        rval = masked_mu + masked_V
        rval.name = 'init_inpainting_state'
        return rval


    def inpaint_update(self, state_above, layer_above, drop_mask, V):

        msg = layer_above.downward_message(state_above)
        mu = self.mu

        z = msg + mu
        z.name = 'inpainting_z_[unknown_iter]'

        if drop_mask is not None:
            rval = drop_mask * z + (1-drop_mask) * V
        else:
            rval = z

        rval.name = 'inpainted_V[unknown_iter]'

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

    def recons_cost(self, V, V_hat, drop_mask = None):

        assert V.ndim == V_hat.ndim
        unmasked_cost = 0.5 * self.beta * T.sqr(V-V_hat) - 0.5*T.log(self.beta / (2*np.pi))
        assert unmasked_cost.ndim == V_hat.ndim

        if drop_mask is None:
            masked_cost = unmasked_cost
        else:
            masked_cost = drop_mask * unmasked_cost

        return masked_cost.mean()

    def upward_state(self, total_state):
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



class ConvMaxPool(SuperDBM_HidLayer):
    def __init__(self,
             output_channels,
            kernel_rows,
            kernel_cols,
            pool_rows,
            pool_cols,
            irange,
            layer_name,
            mirror_weights = False,
            init_bias = 0.,
            border_mode = 'valid'):
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

    def set_input_space(self, space):
        """ Note: this resets parameters!"""
        assert isinstance(space, Conv2DSpace)
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

        self.h_space = Conv2DSpace( shape = (self.h_rows, self.h_cols), nchannels = self.output_channels)
        self.output_space = Conv2DSpace( shape = (self.h_rows / self.pool_rows,
                                                self.h_cols / self.pool_cols),
                                                nchannels = self.output_channels)

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

    def get_lr_scalers(self):
        warnings.warn("get_lr_scalers is hardcoded to 1/(# conv positions)")
        h_rows, h_cols = self.h_space.shape
        num_h = float(h_rows * h_cols)
        return { self.transformer._filters : 1./num_h,
                 self.b : 1. / num_h  }

    def upward_state(self, total_state):
        p,h = total_state
        return p

    def downward_state(self, total_state):
        p,h = total_state
        return h

    def mf_update(self, state_below, state_above, layer_above = None, double_weights = False, iter_name = None):

        # debugging crap, remove when done with it
        if self.layer_name == 'h1':
            assert state_below.name.startswith('h0_p')

        if iter_name is None:
            iter_name = 'anon'

        if state_above is not None:
            assert layer_above is not None
            msg = layer_above.downward_message(state_above)
            msg.name = 'msg_from_'+layer_above.layer_name+'_to_'+self.layer_name+'['+iter_name+']'
        else:
            msg = None
        assert hasattr(state_below,'ndim') and state_below.ndim == 4
        if double_weights:
            state_below = 2. * state_below
            state_below.name = self.layer_name + '_'+iter_name + '_2state'
        z = self.transformer.lmul(state_below) + self.b
        if self.layer_name is not None and iter_name is not None:
            z.name = self.layer_name + '_' + iter_name + '_z'
        p,h = max_pool(z, (self.pool_rows, self.pool_cols), msg)

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

        z = self.transformer.lmul(state_below) + self.b
        p, h, p_sample, h_sample = max_pool(z,
                (self.pool_rows, self.pool_cols), msg, theano_rng)

        return p_sample, h_sample

    def downward_message(self, downward_state):
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

        default_h = self.h_space.get_origin_batch(self.dbm.batch_size) + \
                self.b.get_value()

        default_h_theano = self.h_space.make_theano_batch()

        default_h = default_h.astype(default_h_theano.dtype)

        theano_rng = MRG_RandomStreams(numpy_rng.randint(2 ** 16))

        p_exp, h_exp, p_sample, h_sample = max_pool(
                z = default_h_theano,
                pool_shape = (self.pool_rows, self.pool_cols),
                theano_rng = theano_rng)

        p_state = sharedX( self.output_space.get_origin_batch(
            self.dbm.batch_size))

        h_state = sharedX( default_h)

        t2 = time.time()

        f = function([default_h_theano], updates = {
            p_state : p_sample,
            h_state : h_sample
            })

        t3 = time.time()

        f(default_h)

        t4 = time.time()

        print str(self)+'.make_state took',t4-t1
        print '\tcompose time:',t2-t1
        print '\tcompile time:',t3-t2
        print '\texecute time:',t4-t3

        p_state.name = 'p_sample_shared'
        h_state.name = 'h_sample_shared'

        return p_state, h_state


class Softmax(SuperDBM_HidLayer):
    def __init__(self, n_classes, irange):
        self.__dict__.update(locals())
        del self.self

        self.output_space = VectorSpace(n_classes)
        self.b = sharedX( np.zeros((n_classes,)), name = 'softmax_b')

    def set_input_space(self, space):
        self.input_space = space

        if isinstance(space, Conv2DSpace):
            self.input_dim = space.shape[0] * space.shape[1] * space.nchannels
        else:
            raise NotImplementedError()

        rng = np.random.RandomState([2012,07,25])

        self.W = sharedX( rng.uniform(-self.irange,self.irange, (self.input_dim,self.n_classes)), 'softmax_W' )

        self._params = [ self.b, self.W ]

    def mf_update(self, state_below, state_above):
        if state_above is not None:
            raise NotImplementedError()

        state_below = state_below.reshape( (self.dbm.batch_size, self.input_dim) )

        assert self.W.ndim == 2
        return T.nnet.softmax(T.dot(state_below,self.W)+self.b)

    def downward_message(self, downward_state):
        return T.dot(downward_state, self.W.T)



