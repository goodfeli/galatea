from collections import OrderedDict
from pylearn2.costs.cost import Cost
from pylearn2.costs.cost import FixedVarDescr
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor as T
from theano import config
from pylearn2.utils import make_name
from pylearn2.utils import safe_izip
from pylearn2.utils import safe_zip
from pylearn2 import utils
from pylearn2.models.dbm import flatten
from theano import function
from pylearn2.utils import sharedX
import warnings
import numpy as np
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams

warnings.warn("""Check math on all of super DBM--make sure probabilistic max pooling,
gaussian units, and treating half of v as hidden has all been handled correctly. Also,
write unit tests--that KL decreases, grad of KL is zero, turn the correctness check
in the max pooling into a unit test, etc.""")

class SuperInpaint(Cost):
    def __init__(self,
            monitor_multi_inference = False,
                    mask_gen = None,
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
                    toronto_act_coeffs = None,
                    monitor_each_step = False,
                    use_sum = False
                    ):
        self.__dict__.update(locals())
        del self.self
        #assert not (reweight and reweight_correctly)


    def get_monitoring_channels(self, model, X, Y = None, drop_mask = None, drop_mask_Y = None, **kwargs):

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
        drop_mask = scratch['drop_mask']
        if self.supervised:
            drop_mask_Y = scratch['drop_mask_Y']
            new_drop_mask_Y = scratch['new_drop_mask_Y']

        ii = 0
        for name in ['inpaint_cost', 'l1_act_cost', 'toronto_act_cost',
                'reweighted_act_cost']:
            var = scratch[name]
            if var is not None:
                rval['total_inpaint_cost_term_'+str(ii)+'_'+name] = var
                ii = ii + 1

        if self.monitor_each_step:
            for ii, packed in enumerate(safe_izip(history, new_history)):
                state, new_state = packed
                rval['all_inpaint_costs_after_' + str(ii)] = self.cost_from_states(state,
                        new_state,
                        model, X, Y, drop_mask, drop_mask_Y,
                        new_drop_mask, new_drop_mask_Y)

                if ii > 0:
                    prev_state = history[ii-1]
                    V_hat = state['V_hat']
                    prev_V_hat = prev_state['V_hat']
                    rval['max_pixel_diff[%d]'%ii] = abs(V_hat-prev_V_hat).max()

        final_state = history[-1]

        V_hat = final_state['V_hat']
        err = X - V_hat
        masked_err = err * drop_mask
        sum_sqr_err = T.sqr(masked_err).sum(axis=0)
        recons_count = T.cast(drop_mask.sum(axis=0), 'float32')

        empirical_beta = recons_count / sum_sqr_err
        assert empirical_beta.ndim == 1


        rval['empirical_beta_min'] = empirical_beta.min()
        rval['empirical_beta_mean'] = empirical_beta.mean()
        rval['empirical_beta_max'] = empirical_beta.max()

        layers = model.get_all_layers()
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

            rval['err'] = err

            if self.monitor_multi_inference:
                Y_hat = model.inference_procedure.multi_infer(X)

                argmax = T.argmax(Y_hat,axis=1)
                if argmax.dtype != Y_hat.dtype:
                    argmax = T.cast(argmax, Y_hat.dtype)
                err = T.neq(Y , argmax).mean()
                if err.dtype != Y_hat.dtype:
                    err = T.cast(err, Y_hat.dtype)

                rval['multi_err'] = err

        return rval

    def __call__(self, model, X, Y = None, drop_mask = None, drop_mask_Y = None,
            return_locals = False, include_toronto = True, ** kwargs):

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

        X_space = model.get_input_space()

        if drop_mask is None:
            if self.supervised:
                drop_mask, drop_mask_Y = self.mask_gen(X, Y, X_space=X_space)
            else:
                drop_mask = self.mask_gen(X, X_space=X_space)

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

        total_cost, sublocals = self.cost_from_states(final_state, new_final_state, dbm, X, Y, drop_mask, drop_mask_Y, new_drop_mask, new_drop_mask_Y,
                return_locals=True)
        l1_act_cost = sublocals['l1_act_cost']
        inpaint_cost = sublocals['inpaint_cost']
        reweighted_act_cost = sublocals['reweighted_act_cost']

        if not hasattr(self, 'robustness'):
            self.robustness = None
        if self.robustness is not None:
            inpainting_H_hat = history[-1]['H_hat']
            mf_H_hat = dbm.mf(X, Y=Y)
            if self.supervised:
                inpainting_H_hat = inpainting_H_hat[:-1]
                mf_H_hat = mf_H_hat[:-1]
                for ihh, mhh in safe_izip(flatten(inpainting_H_hat), flatten(mf_H_hat)):
                    total_cost += self.robustness * T.sqr(mhh-ihh).sum()

        if not hasattr(self, 'toronto_act_targets'):
            self.toronto_act_targets = None
        toronto_act_cost = None
        if self.toronto_act_targets is not None and include_toronto:
            toronto_act_cost = 0.
            H_hat = history[-1]['H_hat']
            for s, c, t in zip(H_hat, self.toronto_act_coeffs, self.toronto_act_targets):
                if c == 0.:
                    continue
                s, _ = s
                m = s.mean(axis=0)
                toronto_act_cost += c * T.sqr(m-t).mean()
            total_cost += toronto_act_cost

        if return_locals:
            return locals()

        total_cost.name = 'total_inpaint_cost'

        return total_cost

    def get_fixed_var_descr(self, model, X, Y):

        assert Y is not None

        batch_size = model.batch_size

        drop_mask_X = sharedX(model.get_input_space().get_origin_batch(batch_size))
        drop_mask_X.name = 'drop_mask'

        X_space = model.get_input_space()

        updates = OrderedDict()
        rval = FixedVarDescr()
        inputs=[X, Y]

        if not self.supervised:
            update_X = self.mask_gen(X, X_space = X_space)
        else:
            drop_mask_Y = sharedX(np.ones(batch_size,))
            drop_mask_Y.name = 'drop_mask_Y'
            update_X, update_Y = self.mask_gen(X, Y, X_space)
            updates[drop_mask_Y] = update_Y
            rval.fixed_vars['drop_mask_Y'] =  drop_mask_Y
        if self.mask_gen.sync_channels:
            n = update_X.ndim
            assert n == drop_mask_X.ndim - 1
            update_X.name = 'raw_update_X'
            zeros_like_X = T.zeros_like(X)
            zeros_like_X.name = 'zeros_like_X'
            update_X = zeros_like_X + update_X.dimshuffle(0,1,2,'x')
            update_X.name = 'update_X'
        updates[drop_mask_X] = update_X

        rval.fixed_vars['drop_mask'] = drop_mask_X

        if hasattr(model.inference_procedure, 'V_dropout'):
            include_prob = model.inference_procedure.include_prob
            include_prob_V = model.inference_procedure.include_prob_V
            include_prob_Y = model.inference_procedure.include_prob_Y

            theano_rng = MRG_RandomStreams(2012+11+20)
            for elem in flatten([model.inference_procedure.V_dropout]):
                updates[elem] = theano_rng.binomial(p=include_prob_V, size=elem.shape, dtype=elem.dtype, n=1) / include_prob_V
            if "Softmax" in str(type(model.hidden_layers[-1])):
                hid = model.inference_procedure.H_dropout[:-1]
                y = model.inference_procedure.H_dropout[-1]
                updates[y] = theano_rng.binomial(p=include_prob_Y, size=y.shape, dtype=y.dtype, n=1) / include_prob_Y
            else:
                hid = model.inference_procedure.H_dropout
            for elem in flatten(hid):
                updates[elem] =  theano_rng.binomial(p=include_prob, size=elem.shape, dtype=elem.dtype, n=1) / include_prob

        rval.on_load_batch = [utils.function(inputs, updates=updates)]

        return rval


    def get_gradients(self, model, X, Y = None, **kwargs):

        scratch = self(model, X, Y, include_toronto = False, return_locals=True, **kwargs)

        total_cost = scratch['total_cost']

        params = list(model.get_params())
        grads = dict(safe_zip(params, T.grad(total_cost, params, disconnected_inputs='ignore')))

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

    def get_inpaint_cost(self, dbm, X, V_hat_unmasked, drop_mask, state, Y, drop_mask_Y):
        rval = dbm.visible_layer.recons_cost(X, V_hat_unmasked, drop_mask, use_sum=self.use_sum)

        if self.supervised:
            if self.use_sum:
                scale = 1.
            else:
                scale = 1. / float(dbm.get_input_space().get_total_dimension())
            Y_hat_unmasked = state['Y_hat_unmasked']
            rval = rval + \
                    dbm.hidden_layers[-1].recons_cost(Y, Y_hat_unmasked, drop_mask_Y, scale)

        return rval



    def cost_from_states(self, state, new_state, dbm, X, Y, drop_mask, drop_mask_Y,
            new_drop_mask, new_drop_mask_Y, return_locals = False):

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

        if not hasattr(self, 'use_sum'):
            self.use_sum = False

        inpaint_cost = self.get_inpaint_cost(dbm, X, V_hat_unmasked, drop_mask, state, Y, drop_mask_Y)

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

        if not hasattr(self, 'range_rewards'):
            self.range_rewards = None
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

        if not hasattr(self, 'stdev_rewards'):
            self.stdev_rewards = None
        if self.stdev_rewards is not None:
            assert False # not monitored yet
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

        l1_act_cost = None
        if self.l1_act_targets is not None:
            l1_act_cost = 0.
            if self.l1_act_eps is None:
                self.l1_act_eps = [ None ] * len(self.l1_act_targets)
            for layer, mf_state, targets, coeffs, eps in \
                    safe_izip(dbm.hidden_layers, state['H_hat'] , self.l1_act_targets, self.l1_act_coeffs, self.l1_act_eps):

                assert not isinstance(targets, str)

                try:
                    layer_cost = layer.get_l1_act_cost(mf_state, targets, coeffs, eps)
                except NotImplementedError:
                    if coeffs == 0.:
                        layer_cost = 0.
                    else:
                        raise
                if layer_cost != 0.:
                    l1_act_cost += layer_cost
                # end for substates
            # end for layers
            total_cost += l1_act_cost
        # end if act penalty

        if not hasattr(self, 'hid_presynaptic_cost'):
            self.hid_presynaptic_cost = None
        if self.hid_presynaptic_cost is not None:
            assert False # not monitored yet
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

        if not hasattr(self, 'reweighted_act_targets'):
            self.reweighted_act_targets = None
        reweighted_act_cost = None
        if self.reweighted_act_targets is not None:
            reweighted_act_cost = 0.
            warnings.warn("reweighted_act_cost is hardcoded for sigmoid layers and doesn't check that this is "
                    "what we get.")
            for c, t, s in safe_izip(self.reweighted_act_coeffs, self.reweighted_act_targets, state['H_hat']):
                if c == 0:
                    continue
                s, _ = s
                m = s.mean(axis=0)
                d = T.sqr(m-t)
                weight = 1./(1e-7+s*(1-s))
                reweighted_act_cost += c * (weight * d).mean()
            total_cost += reweighted_act_cost

        total_cost.name = 'total_cost(V_hat_unmasked = %s)' % V_hat_unmasked.name

        if return_locals:
            return total_cost, locals()

        return total_cost


class MaskGen:
    def __init__(self, drop_prob, balance = False, sync_channels = True, drop_prob_y = None, seed = 20120712):
        self.__dict__.update(locals())
        del self.self


    def __call__(self, X, Y = None, X_space=None):
        """
        Note that calling this repeatedly will yield the same random numbers each time.
        """
        assert X_space is not None
        self.called = True
        assert X.dtype == config.floatX
        theano_rng = RandomStreams(self.seed)

        if X.ndim == 2 and self.sync_channels:
            raise NotImplementedError()

        p = self.drop_prob

        if not hasattr(self, 'drop_prob_y') or self.drop_prob_y is None:
            yp = p
        else:
            yp = self.drop_prob_y

        batch_size = X_space.batch_size(X)

        if self.balance:
            flip = theano_rng.binomial(
                    size = (batch_size,),
                    p = 0.5,
                    n = 1,
                    dtype = X.dtype)

            yp = flip * (1-p) + (1-flip) * p

            dimshuffle_args = ['x'] * X.ndim

            if X.ndim == 2:
                dimshuffle_args[0] = 0
                assert not self.sync_channels
            else:
                dimshuffle_args[X_space.axes.index('b')] = 0
                if self.sync_channels:
                    del dimshuffle_args[X_space.axes.index('c')]

            flip = flip.dimshuffle(*dimshuffle_args)

            p = flip * (1-p) + (1-flip) * p

        #size needs to have a fixed length at compile time or the
        #theano random number generator will be angry
        size = tuple([ X.shape[i] for i in xrange(X.ndim) ])
        if self.sync_channels:
            del size[X_space.axes.index('c')]

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
                    size = (batch_size, ),
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
                rval['max_h0_diff[%d]' % ii] = abs(h0[0] - prev_h0[0]).max()

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


class MonitorHack(Cost):
    supervised = True

    def __call__(self,model,  X, Y = None, **kwargs):
        return T.as_tensor_variable(0.)

    def get_monitoring_channels(self, model, X, Y= None, **kwargs):
            rval = OrderedDict()
            Y = T.argmax(Y, axis=1)
            Y = T.cast(Y, X.dtype)

            Y_hat = model.inference_procedure.multi_infer(X)

            argmax = T.argmax(Y_hat,axis=1)
            if argmax.dtype != Y_hat.dtype:
                argmax = T.cast(argmax, Y_hat.dtype)
            err = T.neq(Y , argmax).mean()
            if err.dtype != Y_hat.dtype:
                err = T.cast(err, Y_hat.dtype)

            rval['multi_err'] = err

            return rval
