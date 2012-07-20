from pylearn2.models.model import Model
from pylearn2.space import Conv2DSpace
from pylearn2.utils import sharedX
from pylearn2.linear.conv2d import make_random_conv2D
import theano.tensor as T
import numpy as np
from galatea.dbm.inpaint.probabilistic_max_pooling import max_pool
from theano.gof.op import get_debug_values

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
        rval = self.visible_layer.get_params()
        for layer in self.hidden_layers:
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

    def get_weights(self):
        if len(self.hidden_layers) == 1:
            return self.hidden_layers[0].get_weights()
        else:
            raise NotImplementedError()

    def get_weights_topo(self):
        if len(self.hidden_layers) == 1:
            return self.hidden_layers[0].get_weights_topo()
        else:
            raise NotImplementedError()

    def do_inpainting(self, V, drop_mask, return_history = False):

        history = []

        V_hat = self.visible_layer.init_inpainting_state(V,drop_mask)

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


class SuperDBM_Layer(Model):
    pass

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
        self.beta = sharedX( self.space.get_origin() + init_beta,name = 'beta')
        self.mu = sharedX( self.space.get_origin() + init_mu, name = 'mu')

    def get_params(self):
        return set([self.beta, self.mu])

    def censor_update(self, updates):
        if self.beta in updates:
            updates[self.beta] = T.clip(updates[self.beta],1.,1e6)

    def init_inpainting_state(self, V, drop_mask):

        for Vv, drop_mask_v in get_debug_values(V, drop_mask):
            assert Vv.ndim == 4
            assert drop_mask_v.ndim in [3,4]
            for i in xrange(drop_mask.ndim):
                if Vv.shape[i] != drop_mask_v.shape[i]:
                    print Vv.shape
                    print drop_mask_v.shape
                    assert False

        masked_mu = self.mu * drop_mask
        masked_V  = V  * (1-drop_mask)
        rval = masked_mu + masked_V
        return rval

    def inpaint_update(self, state_above, layer_above, drop_mask, V):

        z = layer_above.downward_message(state_above) + self.mu

        rval = drop_mask * z + (1-drop_mask) * V

        return rval

    def recons_cost(self, V, V_hat, drop_mask = None):

        unmasked_cost = 0.5 * self.beta * T.sqr(V-V_hat) - 0.5*T.log(self.beta / (2*np.pi))

        if drop_mask is None:
            masked_cost = unmasked_cost
        else:
            masked_cost = drop_mask * unmasked_cost

        return masked_cost.mean()

    def upward_state(self, total_state):
        V = total_state
        upward_state = V * self.beta
        return upward_state

class ConvMaxPool(SuperDBM_Layer):
    def __init__(self,
            output_channels,
            kernel_rows,
            kernel_cols,
            pool_rows,
            pool_cols,
            irange,
            layer_name,
            init_bias = 0.):
        self.__dict__.update(locals())
        del self.self

        self.b = sharedX( np.zeros((output_channels,)), name = layer_name + '_b')

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

        self.transformer = make_random_conv2D(self.irange, input_space = space,
                output_space = self.h_space, kernel_shape = (self.kernel_rows, self.kernel_cols),
                batch_size = self.dbm.batch_size)

    def get_params(self):
        return self.transformer.get_params().union([self.b])

    def upward_state(self, total_state):
        p,h = total_state
        return p

    def downward_state(self, total_state):
        p,h = total_state
        return h

    def mf_update(self, state_below, state_above):
        if state_above is not None:
            raise NotImplementedError()
        assert hasattr(state_below,'ndim') and state_below.ndim == 4
        z = self.transformer.lmul(state_below) + self.b
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



