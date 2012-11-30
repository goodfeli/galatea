from collections import OrderedDict
from pylearn2.costs.cost import Cost
from pylearn2.costs.cost import FixedVarDescr
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor as T
from theano import config
from pylearn2.utils import make_name
from pylearn2.utils import safe_izip
from pylearn2.utils import safe_zip
from pylearn2.models.dbm import flatten
from theano import function
from pylearn2.utils import sharedX
import warnings
import numpy as np
from theano.printing import Print

warnings.warn("""Check math on all of super DBM--make sure probabilistic max pooling,
gaussian units, and treating half of v as hidden has all been handled correctly. Also,
write unit tests--that KL decreases, grad of KL is zero, turn the correctness check
in the max pooling into a unit test, etc.""")

class SuperInpaint(Cost):
    def __init__(self,
                    mask_gen,
                    noise = False,
                    both_directions = False,
                    l1_act_coeffs = None,
                    l1_act_targets = None,
                    l1_act_eps = None,
                    range_rewards = None,
                    stdev_rewards = None,
                    robustness = None,
                    supervised = False,
                    niter = None,
                    block_grad = None,
                    vis_presynaptic_cost = None,
                    hid_presynaptic_cost = None,
                    reweighted_act_coeffs = None,
                    reweighted_act_targets = None,
                    toronto_act_targets = None,
                    toronto_act_coeffs = None
                    ):
        self.__dict__.update(locals())
        del self.self
        #assert not (reweight and reweight_correctly)


    def get_monitoring_channels(self, model, X, Y = None, drop_mask = None, drop_mask_Y = None):

        if self.supervised:
            assert Y is not None

        rval = OrderedDict()

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
            inpaint_Y_hat = history[-1]['H_hat'][-1]
            err = T.neq(T.argmax(inpaint_Y_hat, axis=1), T.argmax(Y, axis=1))
            assert err.ndim == 1
            assert drop_mask_Y.ndim == 1
            err =  T.dot(err, drop_mask_Y) / drop_mask_Y.sum()
            if err.dtype != inpaint_Y_hat.dtype:
                err = T.cast(err, inpaint_Y_hat.dtype)

            rval['inpaint_err'] = err

            Y_hat = model.mf(X)[-1]

            Y = T.argmax(Y, axis=1)
            Y = T.cast(Y, Y_hat.dtype)

            argmax = T.argmax(Y_hat,axis=1)
            if argmax.dtype != Y_hat.dtype:
                argmax = T.cast(argmax, Y_hat.dtype)
            err = T.neq(Y , argmax).mean()
            if err.dtype != Y_hat.dtype:
                err = T.cast(err, Y_hat.dtype)

            rval.update(OrderedDict([('err', err)]))

        return rval

    def __call__(self, model, X, Y = None, drop_mask = None, drop_mask_Y = None,
            return_locals = False, include_toronto = True):

        if not self.supervised:
            assert drop_mask_Y is None
            Y = None # ignore Y if some other cost is supervised and has made it get passed in
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
                drop_mask_Y = drop_mask_Y, return_history = True, noise = self.noise,
                niter = self.niter, block_grad = self.block_grad)
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
                    drop_mask_Y = new_drop_mask_Y, return_history = True, noise = self.noise,
                    niter = self.niter, block_grad = self.block_grad)

        new_final_state = new_history[-1]

        total_cost = self.cost_from_states(final_state, new_final_state, dbm, X, Y, drop_mask, drop_mask_Y, new_drop_mask, new_drop_mask_Y)

        if self.robustness is not None:
            inpainting_H_hat = history[-1]['H_hat']
            mf_H_hat = dbm.mf(X, Y=Y)
            if self.supervised:
                inpainting_H_hat = inpainting_H_hat[:-1]
                mf_H_hat = mf_H_hat[:-1]
                for ihh, mhh in safe_izip(flatten(inpainting_H_hat), flatten(mf_H_hat)):
                    total_cost += self.robustness * T.sqr(mhh-ihh).sum()

        if self.toronto_act_targets is not None and include_toronto:
            H_hat = history[-1]['H_hat']
            for s, c, t in zip(H_hat, self.toronto_act_coeffs, self.toronto_act_targets):
                if c == 0.:
                    continue
                s, _ = s
                m = s.mean(axis=0)
                total_cost += c * T.sqr(m-t).mean()

        if return_locals:
            return locals()

        return total_cost

    def get_fixed_var_descr(self, model, X, Y):

        assert Y is not None

        batch_size = model.batch_size

        drop_mask_X = sharedX(model.get_input_space().get_origin_batch(batch_size))
        drop_mask_X.name = 'drop_mask'

        updates = OrderedDict()
        rval = FixedVarDescr()
        inputs=[X, Y]

        if not self.supervised:
            update_X = self.mask_gen(X)
        else:
            drop_mask_Y = sharedX(np.ones(batch_size,))
            drop_mask_Y.name = 'drop_mask_Y'
            update_X, update_Y = self.mask_gen(X, Y)
            updates[drop_mask_Y] = update_Y
            rval.fixed_vars['drop_mask_Y'] =  drop_mask_Y
        updates[drop_mask_X] = update_X

        rval.fixed_vars['drop_mask'] = drop_mask_X
        rval.on_load_batch = [function(inputs, updates=updates, on_unused_input='ignore')]

        return rval


    def get_gradients(self, model, X, Y = None, **kwargs):

        scratch = self(model, X, Y, include_toronto = False, return_locals=True, **kwargs)

        total_cost = scratch['total_cost']

        params = list(model.get_params())
        grads = dict(safe_zip(params, T.grad(total_cost, params)))

        if self.toronto_act_targets is not None:
            H_hat = scratch['history'][-1]['H_hat']
            for i, packed in enumerate(safe_zip(H_hat, self.toronto_act_coeffs, self.toronto_act_targets)):
                s, c, t = packed
                if c == 0.:
                    continue
                s, _ = s
                m = s.mean(axis=0)
                m_cost = c * T.sqr(m-t).mean()
                real_grads = T.grad(m_cost, s)
                if i == 0:
                    below = X
                else:
                    below = H_hat[i-1][0]
                W, = model.hidden_layers[i].transformer.get_params()
                assert W in grads
                b = model.hidden_layers[i].b

                ancestor = T.scalar()
                hack_W = W + ancestor
                hack_b = b + ancestor

                fake_s = T.dot(below, hack_W) + hack_b
                if fake_s.ndim != real_grads.ndim:
                    print fake_s.ndim
                    print real_grads.ndim
                    assert False
                sources = [ (fake_s, real_grads) ]

                fake_grads = T.grad(cost=None, known_grads=dict(sources), wrt=[below, ancestor, hack_W, hack_b])

                grads[W] = grads[W] + fake_grads[2]
                grads[b] = grads[b] + fake_grads[3]


        return grads, OrderedDict()


    def cost_from_states(self, state, new_state, dbm, X, Y, drop_mask, drop_mask_Y,
            new_drop_mask, new_drop_mask_Y):

        if not self.supervised:
            assert drop_mask_Y is None
            assert new_drop_mask_Y is None
        if self.supervised:
            assert drop_mask_Y is not None
            if self.both_directions:
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

        if self.range_rewards is not None:
            for layer, mf_state, coeffs in safe_izip(
                    dbm.hidden_layers,
                    state['H_hat'],
                    self.range_rewards):
                try:
                    layer_cost = layer.get_range_rewards(mf_state, coeffs)
                except NotImplementedError:
                    if coeffs == 0.:
                        layer_cost = 0.
                    else:
                        raise
                if layer_cost != 0.:
                    total_cost += layer_cost

        if self.stdev_rewards is not None:
            for layer, mf_state, coeffs in safe_izip(
                    dbm.hidden_layers,
                    state['H_hat'],
                    self.stdev_rewards):
                try:
                    layer_cost = layer.get_stdev_rewards(mf_state, coeffs)
                except NotImplementedError:
                    if coeffs == 0.:
                        layer_cost = 0.
                    else:
                        raise
                if layer_cost != 0.:
                    total_cost += layer_cost

        if self.l1_act_targets is not None:
            if self.l1_act_eps is None:
                self.l1_act_eps = [ None ] * len(self.l1_act_targets)
            for layer, mf_state, targets, coeffs, eps in safe_izip(dbm.hidden_layers, state['H_hat'] , self.l1_act_targets, self.l1_act_coeffs, self.l1_act_eps):
                assert not isinstance(targets, str)

                try:
                    layer_cost = layer.get_l1_act_cost(mf_state, targets, coeffs, eps)
                except NotImplementedError:
                    if coeffs == 0.:
                        layer_cost = 0.
                    else:
                        raise
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

        if self.hid_presynaptic_cost is not None:
            for c, s, in safe_izip(self.hid_presynaptic_cost, state['H_hat']):
                if c == 0.:
                    continue
                s = s[1]
                assert hasattr(s, 'owner')
                owner = s.owner
                assert owner is not None
                op = owner.op

                if not hasattr(op, 'scalar_op'):
                    raise ValueError("Expected V_hat_unmasked to be generated by an Elemwise op, got "+str(op)+" of type "+str(type(op)))
                assert isinstance(op.scalar_op, T.nnet.sigm.ScalarSigmoid)
                z ,= owner.inputs

                total_cost += c * T.sqr(z).mean()

        if self.reweighted_act_targets is not None:
            # hardcoded for sigmoid layers
            for c, t, s in safe_izip(self.reweighted_act_coeffs, self.reweighted_act_targets, state['H_hat']):
                if c == 0:
                    continue
                s, _ = s
                m = s.mean(axis=0)
                d = T.sqr(m-t)
                weight = 1./(1e-7+s*(1-s))
                total_cost += c * (weight * d).mean()


        total_cost.name = 'total_cost(V_hat_unmasked = %s)' % V_hat_unmasked.name

        return total_cost


class MaskGen:
    def __init__(self, drop_prob, balance, sync_channels = True, drop_prob_y = None):
        self.__dict__.update(locals())
        del self.self

    def __call__(self, X, Y = None):
        if hasattr(self, 'called'):
            # shouldn't be called twice because the seed is hardcoded
            # inside the call.
            # also, for current application, calling twice indicates a bug.
            assert False
        self.called = True
        assert X.dtype == config.floatX
        theano_rng = RandomStreams(20120712)

        if X.ndim == 2 and self.sync_channels:
            raise NotImplementedError()

        p = self.drop_prob

        if self.drop_prob_y is None:
            yp = p
        else:
            yp =self.drop_prob_y

        if self.balance:
            flip = theano_rng.binomial(
                    size = ( X.shape[0] ,),
                    p = 0.5,
                    n = 1,
                    dtype = X.dtype)

            yp = flip * (1-p) + (1-flip) * p

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
            assert isinstance(yp, float) or yp.ndim < 2
            drop_mask_Y = theano_rng.binomial(
                    size = (X.shape[0], ),
                    p = yp,
                    n = 1,
                    dtype = X.dtype)
            assert drop_mask_Y.ndim == 1
            Y_name = make_name(Y, 'anon_Y')
            drop_mask_Y.name = 'drop_mask_Y(%s)' % Y_name
            #drop_mask = Print('drop_mask',attrs=['sum'])(drop_mask)
            #drop_mask_Y = Print('drop_mask_Y',attrs=['sum'])(drop_mask_Y)
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
                    l1_act_targets = None,
                    l1_act_eps = None
                    ):
        self.__dict__.update(locals())
        del self.self
        self.theano_rng = RandomStreams(20120930)


    def get_monitoring_channels(self, model, X, Y = None, drop_mask = None):


        rval = OrderedDict()

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
            new_elem = OrderedDict([( 'V_hat' , V_hat )])
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
            for mf_state, targets, coeffs, eps, layer in safe_izip(state['H_hat'] ,
                    self.l1_act_targets, self.l1_act_coeffs, self.l1_act_eps, dbm.hidden_layers):
                assert not isinstance(targets, str)
                if not isinstance(targets, (list, tuple)):
                    assert not isinstance(mf_state, (list, tuple))
                    mf_state = [ mf_state ]
                    targets = [ targets ]
                    coeffs = [ coeffs ]
                    eps = [ eps ]
                total_cost += layer.get_l1_activation_cost(
                        state = mf_state,
                        targets = targets,
                        coeffs = coeffs,
                        eps = eps)
                # end for substates
            # end for layers
        # end if act penalty

        total_cost.name = 'total_cost(V_hat = %s)' % V_hat.name

        return total_cost

