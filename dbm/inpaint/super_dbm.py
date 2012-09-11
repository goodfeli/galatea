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
        self.hidden_layers[0].set_input_space(visible_layer.space)
        for i in xrange(1,len(hidden_layers)):
            hidden_layers[i].set_input_space(hidden_layers[i-1].get_output_space())
        self.force_batch_size = batch_size

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
        warnings.warn("get_lr_scalers is hardcoded")
        return { self.hidden_layers[0].transformer._filters : 1./(26.**2),
                 self.hidden_layers[0].b : 1. / (26. ** 2)  }

    def get_weights(self):
        if len(self.hidden_layers) == 1:
            return self.hidden_layers[0].get_weights()
        else:
            raise NotImplementedError()

    def get_weights_topo(self):
        return self.hidden_layers[0].get_weights_topo()

    def do_inpainting(self, V, drop_mask, return_history = False, noise = False):

        history = []

        V_hat = self.visible_layer.init_inpainting_state(V,drop_mask,noise)

        H_hat = []
        for i in xrange(0,len(self.hidden_layers)-1):
            #do double weights update for_layer_i
            raise NotImplementedError()
        if len(self.hidden_layers) > 1:
            H_hat.append( self.hidden_layers[-1].mf_update(
                state_above = None,
                layer_above = None,
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
                else:
                    state_above = self.hidden_layers[j+1].downward_state(H_hat[j+1])
                H_hat[j] = self.hidden_layers[j].mf_update(
                        state_below = state_below,
                        state_above = state_above)

            V_hat = self.visible_layer.inpaint_update(
                    state_above = self.hidden_layers[0].downward_state(H_hat[0]),
                    layer_above = self.hidden_layers[0],
                    V = V,
                    drop_mask = drop_mask)

            for j in xrange(1,len(H_hat),2):
                layer_below = self.hidden_layers[j-1].upward_state(H_hat[j-1])
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
                    else:
                        state_above = self.hidden_layers[j+1].downward_state(H_hat[j+1])
                    H_hat[j] = self.hidden_layers[j].mf_update(
                            state_below = state_below,
                            state_above = state_above)

                for j in xrange(1,len(H_hat),2):
                    layer_below = self.hidden_layers[j-1].upward_state(H_hat[j-1])
                    if j == len(H_hat) - 1:
                        state_above = None
                    else:
                        state_above = self.hidden_layers[j+1].downward_state(H_hat[j+1])
                    H_hat[j] = self.hidden_layers[j].mf_update(
                            state_below = state_below,
                            state_above = state_above)
                    #end ifelse
                #end for j
                history.append(H_hat)
            #end for i

        if return_history:
            return history
        else:
            return H_hat

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


class SuperDBM_Layer(Model):

    def upward_state(self, total_state):
        return total_state

    def make_state(self, num_examples, numpy_rng):
        """ Returns a shared variable containing an actual state
           (not a mean field state) for this variable.
        """

        raise NotImplementedError("%s doesn't implement make_state" %
                type(self))

class SuperDBM_HidLayer(SuperDBM_Layer):

    def downward_state(self, total_state):
        return total_state

class GaussianConvolutionalVisLayer(SuperDBM_Layer):
    def __init__(self,
            rows,
            cols,
            channels,
            init_beta,
            init_mu):

        self.__dict__.update(locals())
        del self.self

        self.space = Conv2DSpace(shape = [rows,cols], nchannels = channels)
        self.input_space = self.space
        self.beta = sharedX( self.space.get_origin() + init_beta,name = 'beta')
        self.mu = sharedX( self.space.get_origin() + init_mu, name = 'mu')

    def get_params(self):
        return set([self.beta, self.mu])

    def censor_update(self, updates):
        if self.beta in updates:
            updates[self.beta] = T.clip(updates[self.beta],1.,1e6)

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

        if noise:
            theano_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(42)
            masked_mu += theano_rng.normal(avg = 0.,
                    std = 1., size = masked_mu.shape,
                    dtype = masked_mu.dtype)

        masked_V  = V  * (1-drop_mask)
        rval = masked_mu + masked_V
        return rval

    def inpaint_update(self, state_above, layer_above, drop_mask, V):

        msg = layer_above.downward_message(state_above)
        mu = self.mu

        z = msg + mu

        rval = drop_mask * z + (1-drop_mask) * V

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
        V = total_state
        upward_state = V * self.beta
        return upward_state

    def make_state(self, num_examples, numpy_rng):

        rows, cols = self.space.shape
        channels = self.space.nchannels

        sample = numpy_rng.randn(num_examples, rows, cols, channels)

        sample *= 1./np.sqrt(self.beta.get_value())
        sample += self.mu.get_value()

        rval = sharedX(sample)

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
            init_bias = 0.,
            border_mode = 'valid'):
        self.__dict__.update(locals())
        del self.self

        self.b = sharedX( np.zeros((output_channels,)) + init_bias, name = layer_name + '_b')
        assert border_mode in ['full','valid']

    def set_input_space(self, space):
        assert isinstance(space, Conv2DSpace)
        self.input_space = space
        self.input_rows, self.input_cols = space.shape
        self.input_channels = space.nchannels

        self.h_rows = self.input_rows - self.kernel_rows + 1
        self.h_cols = self.input_cols - self.kernel_cols + 1

        assert self.h_rows % self.pool_rows == 0
        assert self.h_cols % self.pool_cols == 0

        self.h_space = Conv2DSpace( shape = (self.h_rows, self.h_cols), nchannels = self.output_channels)
        self.output_space = Conv2DSpace( shape = (self.h_rows / self.pool_rows,
                                                self.h_cols / self.pool_cols),
                                                nchannels = self.output_channels)

        self.transformer = make_random_conv2D(self.irange, input_space = space,
                output_space = self.h_space, kernel_shape = (self.kernel_rows, self.kernel_cols),
                batch_size = self.dbm.batch_size, border_mode = self.border_mode)
        self.transformer._filters.name = self.layer_name + '_W'
        W ,= self.transformer.get_params()
        assert W.name is not None

    def get_params(self):
        assert self.b.name is not None
        W ,= self.transformer.get_params()
        assert W.name is not None
        return self.transformer.get_params().union([self.b])

    def upward_state(self, total_state):
        p,h = total_state
        return p

    def downward_state(self, total_state):
        p,h = total_state
        return h

    def mf_update(self, state_below, state_above, double_weights = False, iter_name = None):
        if state_above is not None:
            raise NotImplementedError()
        assert hasattr(state_below,'ndim') and state_below.ndim == 4
        if double_weights:
            state_below = 2. * state_below
            state_below.name = self.layer_name + '_'+iter_name + '_2state'
        z = self.transformer.lmul(state_below) + self.b
        if self.layer_name is not None and iter_name is not None:
            z.name = self.layer_name + '_' + iter_name + '_z'
        p,h = max_pool(z, (self.pool_rows, self.pool_cols))

        return p, h

    def downward_message(self, downward_state):
        return self.transformer.lmul_T(downward_state)

    def set_batch_size(self, batch_size):
        self.transformer.set_batch_size(batch_size)

    def get_weights_topo(self):
        outp, inp, rows, cols = range(4)
        raw = self.transformer._filters.get_value()

        return np.transpose(raw,(outp,rows,cols,inp))

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


