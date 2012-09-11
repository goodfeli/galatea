from pylearn2.costs.cost import UnsupervisedCost
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor as T
from theano.printing import Print
import numpy as np
from pylearn2.utils import make_name
import warnings

warnings.warn("""Check math on all of super DBM--make sure probabilistic max pooling,
gaussian units, and treating half of v as hidden has all been handled correctly. Also,
write unit tests--that KL decreases, grad of KL is zero, turn the correctness check
in the max pooling into a unit test, etc.""")

class SuperInpaint(UnsupervisedCost):
    def __init__(self,
                    mask_gen = None,
                    noise = False,
                    both_directions = False
                    #inpaint_penalty = 1.,
                    #weight_decay = None,
                    #balance = False,
                    #reweight = True,
                    #reweight_correctly = False,
                    #recons_penalty = None,
                    #h_target = None,
                    #h_penalty = None,
                    #g_target = None,
                    #g_penalty = None,
                    #h_bimodality_penalty = None,
                    #g_bimodality_penalty = None,
                    #h_contractive_penalty = None,
                    #g_contractive_penalty = None
                    ):
        self.__dict__.update(locals())
        del self.self
        #assert not (reweight and reweight_correctly)


    def get_monitoring_channels(self, model, X, Y = None, drop_mask = None):

        rval = {}

        """
        d = self(model, X = X, drop_mask = drop_mask, return_locals = True)

        H_hat = d['H_hat']
        G_hat = d['G_hat']

        h = H_hat.mean(axis=0)
        g = G_hat.mean(axis=0)


        rval['h_min'] = h.min()
        rval['h_mean'] = h.mean()
        rval['h_max'] = h.max()

        rval['g_min'] = g.min()
        rval['g_mean'] = g.mean()
        rval['g_max'] = g.max()

        h_max = H_hat.max(axis=0)
        h_min = H_hat.min(axis=0)
        g_max = G_hat.max(axis=0)
        g_min = G_hat.min(axis=0)

        h_range = h_max - h_min
        g_range = g_max - g_min

        rval['h_range_min'] = h_range.min()
        rval['h_range_mean'] = h_range.mean()
        rval['h_range_max'] = h_range.max()

        rval['g_range_min'] = g_range.min()
        rval['g_range_mean'] = g_range.mean()
        rval['g_range_max'] = g_range.max()
        """

        hist = self(model, X, drop_mask, return_locals = True)['history']

        for ii, state in enumerate(hist):
            rval['obj_after_' + str(ii)] = self.cost_from_state(state,
                    model, X, drop_mask)

        return rval

    def __call__(self, model, X, drop_mask = None, return_locals = False):

        if not hasattr(model,'cost'):
            model.cost = self
        if not hasattr(model,'mask_gen'):
            model.mask_gen = self.mask_gen

        dbm = model

        if drop_mask is None:
            drop_mask = self.mask_gen(X)
        else:
            assert self.mask_gen is None

        if drop_mask.ndim < X.ndim:
            if self.mask_gen is not None:
                assert self.mask_gen.sync_channels
            if X.ndim != 4:
                raise NotImplementedError()
            drop_mask = drop_mask.dimshuffle(0,1,2,'x')

        if not hasattr(self,'noise'):
            self.noise = False

        history = dbm.do_inpainting(X, drop_mask, return_history = True, noise = self.noise)

        final_state = history[-1]

        total_cost = self.cost_from_state(final_state, dbm, X, drop_mask)

        if return_locals:
            return locals()

        return total_cost

    def cost_from_state(self, state, dbm, X, drop_mask):

        V_hat = state['V_hat']

        inpaint_cost = dbm.visible_layer.recons_cost(X, V_hat, drop_mask)

        if not hasattr(self, 'both_directions'):
            self.both_directions = False

        if self.both_directions:
            new_drop_mask = 1. - drop_mask

            new_history = dbm.do_inpainting(X, new_drop_mask, return_history = True, noise = self.noise)

            new_final_state = new_history[-1]

            new_V_hat = new_final_state['V_hat']

            new_inpaint_cost = dbm.visible_layer.recons_cost(X, new_V_hat, new_drop_mask)
            inpaint_cost = 0.5 * inpaint_cost + 0.5 * new_inpaint_cost

        total_cost = inpaint_cost


        return total_cost


class MaskGen:
    def __init__(self, drop_prob, balance, sync_channels = True):
        self.__dict__.update(locals())
        del self.self

    def __call__(self, X):
        assert X.dtype == 'float32'
        theano_rng = RandomStreams(20120712)

        if X.ndim == 2 and self.sync_channels:
            raise NotImplementedError()

        #size needs to have a fixed length at compile time or the
        #theano random number generator will be angry
        size = tuple([ X.shape[i] for i in xrange(X.ndim) ])

        if self.sync_channels:
            size = size[:-1]

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


        drop_mask = theano_rng.binomial(
                    size = size,
                    p = p,
                    n = 1,
                    dtype = X.dtype)

        X_name = make_name(X, 'anon_X')
        drop_mask.name = 'drop_mask(%s)' % X_name

        return drop_mask
