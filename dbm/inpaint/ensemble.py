from theano import tensor as T

from pylearn2.models.model import Model
from pylearn2.utils import sharedX

from galatea.dbm.inpaint.super_inpaint import SuperInpaint

class EnsembleDBM(Model):

    def __init__(self, dbm, num_copies):
        self.__dict__.update(locals())
        del self.self

        self.ensemble_params = []

        self.inference_procedure = EnsembleInferenceProcedure(self)

        for i in xrange(num_copies):
            params = self.get_params()
            self.ensemble_params.insert(0, dbm.get_params())
            for j in xrange(len(self.ensemble_params[i])):
                if self.ensemble_params[i][j] in params:
                    self.ensemble_params[i][j] = sharedX(self.ensemble_params[i][j].get_value(), self.ensemble_params[i][j].name)
                self.ensemble_params[i][j].name = self.ensemble_params[i][j].name + '_' + str(j)
            if i < num_copies - 1:
                self.dbm._update_layer_input_spaces()

        self.batch_size = dbm.batch_size
        self.force_batch_size = dbm.batch_size

    def get_all_layers(self):
        return self.dbm.get_all_layers()

    def get_output_space(self):
        return self.dbm.get_output_space()

    def get_ensemble_variants(self, param):
        ensemble_params = self.ensemble_params
        j = ensemble_params[0].index(param)
        return [member_params[j] for member_params in ensemble_params]

    def get_params(self):
        return self.dbm.get_params()

    def get_input_space(self):
        return self.dbm.get_input_space()

    def do_inpainting(self, X, Y = None, drop_mask = None,
                drop_mask_Y = None, return_history = False, noise = False,
                niter = None, block_grad = None):

        leftmost_history = self.dbm.do_inpainting(
                X, Y=Y, drop_mask=drop_mask, drop_mask_Y=drop_mask_Y,
                return_history=True, noise=noise,
                niter=niter, block_grad=block_grad)

        leftmost = leftmost_history[-1]

        H_hat = leftmost['H_hat']

        V_hat_unmasked = leftmost['V_hat']

        V_hat_unmasked = self.dbm.visible_layer.ensemble_prediction(
                symbolic=V_hat_unmasked, outputs_dict=self.dbm.presynaptic_outputs,
                ensemble=self)

        V_hat = drop_mask * V_hat_unmasked

        if Y is not None:
            Y_hat_unmasked = leftmost['Y_hat_unmasked']
            Y_hat_unmasked = self.dbm.hidden_layers[-1].ensemble_prediction(
                    symbolic=Y_hat_unmasked, outputs_dict=self.dbm.presynaptic_outputs,
                    ensemble=self)
            Y_hat = drop_mask_Y * Y_hat_unmasked


        if return_history:
            # Note: only spoofed history, just includes final step
            # H_hat is only for leftmost member of ensemble
            d = {'V_hat' : V_hat, 'H_hat': list(H_hat), 'V_hat_unmasked': V_hat_unmasked}
            if Y is not None:
                d['Y_hat_unmasked'] = Y_hat_unmasked
                d['Y_hat'] = H_hat[-1]
            return [d]
        else:
            if Y is not None:
                return V_hat, Y_hat
            return V_hat

    def mf(self, V, Y = None, return_history = False, niter = None, block_grad = None):

        drop_mask = T.zeros_like(V)

        if Y is not None:
            drop_mask_Y = T.zeros_like(Y)
        else:
            batch_size = V.shape[0]
            num_classes = self.dbm.hidden_layers[-1].n_classes
            assert isinstance(num_classes, int)
            Y = T.alloc(1., V.shape[0], num_classes)
            drop_mask_Y = T.alloc(1., V.shape[0])

        history = self.do_inpainting(X=V,
            Y=Y,
            return_history=True,
            drop_mask=drop_mask,
            drop_mask_Y=drop_mask_Y,
            noise=False,
            niter=niter,
            block_grad=block_grad)

        if return_history:
            return [elem['H_hat'] for elem in history]

        return history[-1]['H_hat']


class EnsembleSuperInpaint(SuperInpaint):

    def get_inpaint_cost(self, ensemble, X, V_hat_unmasked, drop_mask, state, Y, drop_mask_Y):
        rval = ensemble.dbm.visible_layer.ensemble_recons_cost(X, V_hat_unmasked, drop_mask, use_sum=self.use_sum,
                ensemble=ensemble)

        if self.supervised:
            if self.use_sum:
                scale = 1.
            else:
                scale = 1. / float(ensemble.get_input_space().get_total_dimension())
            Y_hat_unmasked = state['Y_hat_unmasked']
            rval = rval + \
                    ensemble.dbm.hidden_layers[-1].ensemble_recons_cost(Y, Y_hat_unmasked, drop_mask_Y, scale,
                            ensemble=ensemble)

        return rval


    def get_fixed_var_descr(self, model, X, Y):

        if not self.supervised:
            raise NotImplementedError()

        rval = super(EnsembleSuperInpaint, self).get_fixed_var_descr(model, X, Y)

        model.dbm.make_presynaptic_outputs(self.supervised)

        rval.on_load_batch.append(OnEnsembleLoadBatch(model))

        return rval

class OnEnsembleLoadBatch(object):
    """
    A callable object that prepares the presynaptic predictions for all but one
    model in the ensemble and loads that one model into ensemble.dbm
    """

    def __init__(self, ensemble):
        self.__dict__.update(locals())
        del self.self

        assert isinstance(ensemble, EnsembleDBM)

    def __call__(self, X, Y):

        raise NotImplementedError("TODO: clear output")
        raise NotImplementedError("TODO: select random model to be last.")
        raise NotImplementedError("TODO: iteratively load all other models and have them contribute to output.")
        raise NotImplementedError("TODO: load last model.")

class EnsembleInferenceProcedure(object):

    def __init__(self, ensemble):
        self.__dict__.update(locals())
        del self.self

