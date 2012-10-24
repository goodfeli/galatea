from pylearn2.costs.cost import Cost
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor as T
from theano.printing import Print
import numpy as np
from pylearn2.utils import make_name
from pylearn2.utils import safe_izip
import warnings

warnings.warn("""Check math on all of super DBM--make sure probabilistic max pooling,
gaussian units, and treating half of v as hidden has all been handled correctly. Also,
write unit tests--that KL decreases, grad of KL is zero, turn the correctness check
in the max pooling into a unit test, etc.""")

class SuperInpaint(Cost):
    def __init__(self,
                    mask_gen = None,
                    noise = False,
                    both_directions = False,
                    l1_act_coeffs = None,
                    l1_act_targets = None,
                    supervised = False
                    ):
        self.__dict__.update(locals())
        del self.self
        #assert not (reweight and reweight_correctly)


    def get_monitoring_channels(self, model, X, Y = None, drop_mask = None, drop_mask_Y = None):

        if self.supervised:
            assert Y is not None

        rval = {}

        # TODO: shouldn't self() handle this?
        if drop_mask is not None and drop_mask.ndim < X.ndim:
            if self.mask_gen is not None:
                assert self.mask_gen.sync_channels
            if X.ndim != 4:
                raise NotImplementedError()
            drop_mask = drop_mask.dimshuffle(0,1,2,'x')

        scratch = self(model, X, Y, drop_mask = drop_mask, drop_mask_Y = drop_mask_Y,
                return_locals = True)

        history = scratch['history']
        new_history = scratch['new_history']
        new_drop_mask = scratch['new_drop_mask']
        new_drop_mask_Y = None
        if self.supervised:
            drop_mask_Y = scratch['drop_mask_Y']
            new_drop_mask_Y = scratch['new_drop_mask_Y']

        for ii, packed in enumerate(safe_izip(history, new_history)):
            state, new_state = packed
            rval['inpaint_after_' + str(ii)] = self.cost_from_states(state,
                    new_state,
                    model, X, Y, drop_mask, drop_mask_Y,
                    new_drop_mask, new_drop_mask_Y)

            if ii > 0:
                prev_state = history[ii-1]
                V_hat = state['V_hat']
                prev_V_hat = prev_state['V_hat']
                rval['max_pixel_diff[%d]'%ii] = abs(V_hat-prev_V_hat).max()

        final_state = history[-1]

        layers = [ model.visible_layer ] + model.hidden_layers
        states = [ final_state['V_hat'] ] + final_state['H_hat']

        for layer, state in safe_izip(layers, states):
            d = layer.get_monitoring_channels_from_state(state)
            for key in d:
                mod_key = 'final_inpaint_' + layer.layer_name + '_' + key
                assert mod_key not in rval
                rval[mod_key] = d[key]

        if self.supervised:
            Y_hat = model.mf(X)[-1]

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


        return rval

    def __call__(self, model, X, Y = None, drop_mask = None, drop_mask_Y = None,
            return_locals = False):

        if not self.supervised:
            assert drop_mask_Y is None
        if self.supervised:
            assert Y is not None
            if drop_mask is not None:
                assert drop_mask_Y is not None

        if not hasattr(model,'cost'):
            model.cost = self
        if not hasattr(model,'mask_gen'):
            model.mask_gen = self.mask_gen

        dbm = model

        if drop_mask is None:
            if self.supervised:
                drop_mask, drop_mask_Y = self.mask_gen(X, Y)
            else:
                drop_mask = self.mask_gen(X)
        else:
            assert self.mask_gen is None

        if drop_mask_Y is not None:
            assert drop_mask_Y.ndim == 1

        if drop_mask.ndim < X.ndim:
            if self.mask_gen is not None:
                assert self.mask_gen.sync_channels
            if X.ndim != 4:
                raise NotImplementedError()
            drop_mask = drop_mask.dimshuffle(0,1,2,'x')

        if not hasattr(self,'noise'):
            self.noise = False

        history = dbm.do_inpainting(X, Y = Y, drop_mask = drop_mask,
                drop_mask_Y = drop_mask_Y, return_history = True, noise = self.noise)
        final_state = history[-1]

        new_drop_mask = None
        new_drop_mask_Y = None
        new_history = [ None for state in history ]

        if not hasattr(self, 'both_directions'):
            self.both_directions = False
        if self.both_directions:
            new_drop_mask = 1. - drop_mask
            if self.supervised:
                new_drop_mask_Y = 1. - drop_mask_Y
            new_history = dbm.do_inpainting(X, Y = Y, drop_mask = new_drop_mask,
                    drop_mask_Y = new_drop_mask_Y, return_history = True, noise = self.noise)

        new_final_state = new_history[-1]

        total_cost = self.cost_from_states(final_state, new_final_state, dbm, X, Y, drop_mask, drop_mask_Y, new_drop_mask, new_drop_mask_Y)

        if return_locals:
            return locals()

        return total_cost

    def cost_from_states(self, state, new_state, dbm, X, Y, drop_mask, drop_mask_Y,
            new_drop_mask, new_drop_mask_Y):

        if not self.supervised:
            assert drop_mask_Y is None
            assert new_drop_mask_Y is None
        if self.supervised:
            assert drop_mask_Y is not None
            assert new_drop_mask_Y is not None
            assert Y is not None

        V_hat_unmasked = state['V_hat_unmasked']
        assert V_hat_unmasked.ndim == X.ndim

        inpaint_cost = dbm.visible_layer.recons_cost(X, V_hat_unmasked, drop_mask)

        if self.supervised:
            scale = 1. / float(dbm.get_input_space().get_total_dimension())
            Y_hat_unmasked = state['Y_hat_unmasked']
            inpaint_cost = inpaint_cost + \
                    dbm.hidden_layers[-1].recons_cost(Y, Y_hat_unmasked, drop_mask_Y, scale)

        if not hasattr(self, 'both_directions'):
            self.both_directions = False

        assert self.both_directions == (new_state is not None)

        if new_state is not None:

            new_V_hat_unmasked = new_state['V_hat_unmasked']

            new_inpaint_cost = dbm.visible_layer.recons_cost(X, new_V_hat_unmasked, new_drop_mask)
            if self.supervised:
                new_Y_hat_unmasked = new_state['Y_hat_unmasked']
                new_inpaint_cost = new_inpaint_cost + \
                        dbm.hidden_layers[-1].recons_cost(Y, new_Y_hat_unmasked, new_drop_mask_Y, scale)
            # end if include_Y
            inpaint_cost = 0.5 * inpaint_cost + 0.5 * new_inpaint_cost
        # end if both directions

        total_cost = inpaint_cost

        if self.l1_act_targets is not None:
            for layer, mf_state, targets, coeffs in safe_izip(dbm.hidden_layers, state['H_hat'] , self.l1_act_targets, self.l1_act_coeffs):
                assert not isinstance(targets, str)

                layer_cost = layer.get_l1_act_cost(mf_state, targets, coeffs)
                if layer_cost != 0.:
                    total_cost += layer_cost
                #for H, t, c in zip(mf_state, targets, coeffs):
                    #if c == 0.:
                    #    continue
                    #axes = (0,2,3) # all but channel axis
                                  # this assumes bc01 format
                    #h = H.mean(axis=axes)
                    #assert h.ndim == 1
                    #total_cost += c * abs(h - t).mean()
                # end for substates
            # end for layers
        # end if act penalty

        total_cost.name = 'total_cost(V_hat_unmasked = %s)' % V_hat_unmasked.name

        return total_cost


class MaskGen:
    def __init__(self, drop_prob, balance, sync_channels = True):
        self.__dict__.update(locals())
        del self.self

    def __call__(self, X, Y = None):
        assert X.dtype == 'float32'
        theano_rng = RandomStreams(20120712)

        if X.ndim == 2 and self.sync_channels:
            raise NotImplementedError()

        p = self.drop_prob

        if self.balance:
            flip = theano_rng.binomial(
                    size = ( X.shape[0] ,),
                    p = 0.5,
                    n = 1,
                    dtype = X.dtype)

            dimshuffle_args = [ 0 ] + [ 'x' ] * (X.ndim -1 - self.sync_channels)

            flip = flip.dimshuffle(*dimshuffle_args)

            p = flip * (1-p) + (1-flip) * p

        #size needs to have a fixed length at compile time or the
        #theano random number generator will be angry
        size = tuple([ X.shape[i] for i in xrange(X.ndim) ])
        if self.sync_channels:
            size = size[:-1]

        drop_mask = theano_rng.binomial(
                    size = size,
                    p = p,
                    n = 1,
                    dtype = X.dtype)

        X_name = make_name(X, 'anon_X')
        drop_mask.name = 'drop_mask(%s)' % X_name

        if Y is not None:
            drop_mask_Y = theano_rng.binomial(
                    size = (X.shape[0], ),
                    p = p,
                    n = 1,
                    dtype = X.dtype)
            Y_name = make_name(Y, 'anon_Y')
            drop_mask_Y.name = 'drop_mask_Y(%s)' % Y_name
            return drop_mask, drop_mask_Y

        return drop_mask



class SuperDenoise(Cost):
    """
        Implements Score Matching Denoising for RBMs
        For other models, it is an approximation, based on computing
        E[model.hidden_layers[0]] in the mean field distribution,
        instead of the true distribution.
        It is not clear whether it is a good idea to do this, ie, I
        haven't thought through whether it has a problem like what
        happens when you approximate the partition function with mean
        field.
    """
    def __init__(self,
                    noise_precision = 1.,
                    l1_act_coeffs = None,
                    l1_act_targets = None
                    ):
        self.__dict__.update(locals())
        del self.self
        self.theano_rng = RandomStreams(20120930)


    def get_monitoring_channels(self, model, X, Y = None, drop_mask = None):


        rval = {}

        scratch = self(model, X, drop_mask, return_locals = True)

        history = scratch['history']
        X_tilde = scratch['X_tilde']

        for ii, state in enumerate(history):
            rval['obj_after_' + str(ii)] = self.cost_from_state(state,
                    model, X, X_tilde)

            if ii > 0:
                prev_state = history[ii-1]
                V_hat = state['V_hat']
                prev_V_hat = prev_state['V_hat']
                assert V_hat is not prev_V_hat
                rval['max_pixel_diff[%d]'%ii] = abs(V_hat-prev_V_hat).max()
                h0 = state['H_hat'][0]
                prev_h0 = prev_state['H_hat'][0]
                assert h0 is not prev_h0
                rval['max_h0_diff[%d]'%ii] = abs(h0[0] - prev_h0[0]).max()

        final_state = history[-1]

        layers = [ model.visible_layer ] + model.hidden_layers
        states = [ final_state['V_hat'] ] + final_state['H_hat']

        for layer, state in safe_izip(layers, states):
            d = layer.get_monitoring_channels_from_state(state)
            for key in d:
                mod_key = 'final_denoise_' + layer.layer_name + '_' + key
                assert mod_key not in rval
                rval[mod_key] = d[key]

        return rval

    def __call__(self, model, X, drop_mask = None, return_locals = False):

        if not hasattr(model,'cost'):
            model.cost = self
        if not hasattr(model,'mask_gen'):
            model.mask_gen = self.mask_gen

        dbm = model

        X_tilde = self.theano_rng.normal(avg=X, std=1./T.sqrt(self.noise_precision),
                size=X.shape, dtype=X.dtype)

        history = dbm.mf(X_tilde, return_history=True)

        hid = dbm.hidden_layers[0]

        new_history = []
        for elem in history:
            H1 = elem[0]
            ds = hid.downward_state(H1)
            V_hat = dbm.visible_layer.inpaint_update(layer_above = hid, state_above = ds)
            new_elem = { 'V_hat' : V_hat }
            new_elem['H_hat'] = elem
            new_history.append(new_elem)
        history = new_history
        del new_history

        final_state = history[-1]

        new_drop_mask = None

        total_cost = self.cost_from_state(final_state, dbm, X, X_tilde)

        if return_locals:
            return locals()

        return total_cost

    def cost_from_state(self, state, dbm, X, X_tilde):

        V_hat = state['V_hat']

        beta = dbm.visible_layer.beta

        model_term = beta * (X_tilde-V_hat)
        noise_term = self.noise_precision * (X_tilde-X)
        diff = model_term - noise_term
        assert diff.ndim == 4
        smd_cost = T.sqr(diff).sum(axis=(1,2,3)).mean()
        assert smd_cost.ndim == 0

        if not hasattr(self, 'both_directions'):
            self.both_directions = False

        total_cost = smd_cost

        if self.l1_act_targets is not None:
            for mf_state, targets, coeffs in safe_izip(state['H_hat'] , self.l1_act_targets, self.l1_act_coeffs):
                assert not isinstance(targets, str)
                if not isinstance(targets, (list, tuple)):
                    assert not isinstance(mf_state, (list, tuple))
                    mf_state = [ mf_state ]
                    targets = [ targets ]
                    coeffs = [ coeffs ]

                for H, t, c in safe_izip(mf_state, targets, coeffs):
                    if c == 0.:
                        continue
                    axes = (0,2,3) # all but channel axis
                                  # this assumes bc01 format
                    h = H.mean(axis=axes)
                    assert h.ndim == 1
                    total_cost += c * abs(h - t).mean()
                # end for substates
            # end for layers
        # end if act penalty

        total_cost.name = 'total_cost(V_hat = %s)' % V_hat.name

        return total_cost

