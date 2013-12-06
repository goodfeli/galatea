from collections import OrderedDict
from pylearn2.costs.cost import Cost
from pylearn2.costs.cost import FixedVarDescr
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

from pylearn2.costs.dbm import MultiPrediction as SuperInpaint
from pylearn2.costs.dbm import MaskGen

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
