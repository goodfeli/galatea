__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2011, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

import time
from pylearn2.expr.nnet import sigmoid_numpy
from pylearn2.models.model import Model
from theano import config, function
import theano.tensor as T
import numpy as np
import warnings
from theano.gof.op import get_debug_values, debug_error_message
from pylearn2.utils import sharedX, as_floatX
import theano
import gc

warnings.warn('pddbm changing the recursion limit')
import sys
sys.setrecursionlimit(50000)

from pylearn2.models.s3c import full_min
from pylearn2.models.s3c import full_max
from pylearn2.models.s3c import reflection_clip
from pylearn2.models.s3c import damp
from pylearn2.models.s3c import S3C
from pylearn2.models.s3c import SufficientStatistics
from galatea.pddbm.batch_gradient_inference_monitor_hack import BatchGradientInferenceMonitorHack
from theano.printing import min_informative_str
from theano.printing import Print
from theano.gof.op import debug_assert
from theano.gof.op import get_debug_values
from pylearn2.space import VectorSpace
from theano.sandbox.scan import scan

warnings.warn('There is a known bug where for some reason the w field of s3c '
'gets serialized. Not sure if other things get serialized too but be sure to '
'call make_pseudoparams on everything you unpickle. I wish theano was not '
'such a piece of shit!')

def flatten(collection):
    rval = set([])

    for elem in collection:
        if hasattr(elem,'__len__'):
            rval = rval.union(flatten(elem))
        else:
            rval = rval.union([elem])

    return rval


class PDDBM(Model):

    """ Implements a model of the form
        P(v,s,h,g[0],...,g[N_g]) = S3C(v,s|h)DBM(h,g)
    """

    def __init__(self,
            s3c,
            dbm,
            bayes_B = False,
            ss_init_h = None,
            ss_init_scale = None,
            ss_init_mu = None,
            exhaustive_iteration = False,
            s3c_mu_learning_rate_scale = 1.,
            monitor_ranges = False,
            use_diagonal_natural_gradient = False,
            inference_procedure = None,
            learning_rate = 1e-3,
            init_non_s3c_lr = 1e-3,
            non_s3c_lr_start = 0,
            final_non_s3c_lr = 1e-3,
            min_shrink = .05,
            shrink_start = 0,
            lr_shrink_example_scale = 0.,
            non_s3c_lr_saturation_example = None,
            h_penalty = 0.0,
            s3c_l1_weight_decay = 0.0,
            h_target = None,
            g_penalties = None,
            g_targets = None,
            print_interval = 10000,
            freeze_s3c_params = False,
            freeze_dbm_params = False,
            dbm_weight_decay = None,
            dbm_l1_weight_decay = None,
            recons_penalty = None,
            sub_batch = False,
            init_momentum = None,
            final_momentum = None,
            momentum_saturation_example = None,
            h_bias_src = 'dbm',
            monitor_neg_chain_marginals = False):
        """
            s3c: a galatea.s3c.s3c.S3C object
                will become owned by the PDDBM
                it won't be deleted but many of its fields will change
            dbm: a pylearn2.dbm.DBM object
                will become owned by the PDDBM
                it won't be deleted but many of its fields will change
            inference_procedure: a galatea.pddbm.pddbm.InferenceProcedure
                                    if None, does not compile a learn_func
            print_interval: number of examples between each status printout
            h_bias_src: either 'dbm' or 's3c'. both the dbm and s3c have a bias
                    term on h-- whose should we use when we build the model?
                        if you want to start training with a new rbm on top
                        of an s3c model, it probably makes sense to make the
                        rbm have very small weights and take the biases from
                        s3c.
                      if you've already pretrained the rbm, then since this
                      model is basically P(g,h)P(v,s|h) it might make sense
                      to take the biases from the RBM
            freeze_dbm_params: If True, do not update parameters that are owned
                    exclusively by the dbm. (i.e., s3c.bias_hid will still be
                    updated, unless you also freeze_s3c_params)
            freeze_s3c_params: If True, do not update parameters that are owned
                    exclusively by s3c. (i.e., dbm.bias_vis will still be updated, unless you also freeze_dbm_params)
        """

        super(PDDBM,self).__init__()

        self.bayes_B = bayes_B

        self.exhaustive_iteration = exhaustive_iteration
        if self.exhaustive_iteration:
            self.iterator = None

        self.ss_init_h = ss_init_h
        self.ss_init_mu = ss_init_mu
        self.ss_init_scale = ss_init_scale

        self.min_shrink = np.cast['float32'](float(min_shrink))
        self.lr_shrink_example_scale = np.cast['float32'](float(lr_shrink_example_scale))
        self.shrink_start = shrink_start

        self.monitor_ranges = monitor_ranges

        self.learning_rate = learning_rate

        use_cd = dbm.use_cd
        self.use_cd = use_cd

        self.monitor_neg_chain_marginals = monitor_neg_chain_marginals

        self.dbm_weight_decay = dbm_weight_decay
        self.dbm_l1_weight_decay = dbm_l1_weight_decay
        self.use_diagonal_natural_gradient = use_diagonal_natural_gradient

        self.s3c_mu_learning_rate_scale = s3c_mu_learning_rate_scale

        self.init_non_s3c_lr = init_non_s3c_lr
        self.final_non_s3c_lr = final_non_s3c_lr
        self.non_s3c_lr_start = non_s3c_lr_start
        self.non_s3c_lr_saturation_example = non_s3c_lr_saturation_example

        self.init_momentum =   init_momentum
        self.final_momentum = final_momentum
        self.momentum_saturation_example = momentum_saturation_example

        self.recons_penalty = recons_penalty

        self.s3c = s3c
        s3c.e_step.autonomous = False

        if self.s3c.m_step is not None:
            m_step = self.s3c.m_step
            self.B_learning_rate_scale = m_step.B_learning_rate_scale
            self.alpha_learning_rate_scale = m_step.alpha_learning_rate_scale
            self.s3c_W_learning_rate_scale = m_step.W_learning_rate_scale
            if m_step.p_penalty is not None and m_step.p_penalty != 0.0:
                raise ValueError("s3c.p_penalty must be none or 0. p is not tractable anymore "
                        "when s3c is integrated into a pd-dbm.")
            self.B_penalty = m_step.B_penalty
            self.alpha_penalty = m_step.alpha_penalty
        else:
            self.B_learning_rate_scale = 1.
            self.alpha_learning_rate_scale = 1.
            self.s3c_W_learning_rate_scale = 1.
            self.B_penalty = 0.
            self.alpha_penalty = 0.

        s3c.m_step = None
        self.dbm = dbm
        self.dbm.use_cd = use_cd

        self.rng = np.random.RandomState([1,2,3])

        W = dbm.W[0].get_value()

        if ss_init_h is not None:
            W *= 0.
            W += (self.rng.uniform(0.,1., W.shape) < ss_init_h) * (ss_init_mu + self.rng.randn( * W.shape ) * ss_init_scale)
            dbm.W[0].set_value(W)


        if dbm.bias_vis.get_value(borrow=True).shape \
                != s3c.bias_hid.get_value(borrow=True).shape:
                    raise AssertionError("DBM has "+str(dbm.bias_vis.get_value(borrow=True).shape)+\
                            " visible units but S3C has "+str(s3c.bias_hid.get_value(borrow=True).shape))
        if h_bias_src == 'dbm':
            self.s3c.bias_hid = self.dbm.bias_vis
        elif h_bias_src == 's3c':
            self.dbm.bias_vis = self.s3c.bias_hid
        else:
            assert False

        self.nvis = s3c.nvis
        self.input_space = VectorSpace(self.nvis)

        self.freeze_s3c_params = freeze_s3c_params
        self.freeze_dbm_params = freeze_dbm_params

        #don't support some exotic options on s3c
        for option in ['monitor_functional',
                       'recycle_q',
                       'debug_m_step']:
            if getattr(s3c, option):
                warnings.warn('PDDBM does not support '+option+' in '
                        'the S3C layer, disabling it')
                setattr(s3c, option, False)

        s3c.monitor_stats = []
        s3c.e_step.monitor_stats = []

        if s3c.e_step.monitor_kl:
            s3c.e_step.monitor_kl = False

        self.print_interval = print_interval

        s3c.print_interval = None
        dbm.print_interval = None


        if inference_procedure is not None:
            inference_procedure.register_model(self)
        self.inference_procedure = inference_procedure

        self.num_g = len(self.dbm.W)

        self.h_penalty = h_penalty
        self.h_target = h_target

        self.g_penalties = g_penalties
        self.g_targets = g_targets

        self.s3c_l1_weight_decay = s3c_l1_weight_decay

        self.sub_batch = sub_batch

        self.redo_everything()

    def get_weights(self):
        x = input('which weights you want?')

        if x == 2:
            """
            W1 = self.s3c.W.get_value()
            mu = self.s3c.mu.get_value()
            W1s = W1 * mu
            W2 = self.dbm.W[0].get_value()

            H = sigmoid_numpy( W2.T + self.dbm.bias_vis.get_value())

            rval = np.dot(H, W1s.T)
            assert rval.shape[0] == self.s3c.nhid
            assert rval.shape[1] == self.s3c.nvis

            return rval.T
            """

            return np.dot(self.s3c.W.get_value() \
                    * self.s3c.mu.get_value(), self.dbm.W[0].get_value())

        if x == 1:
            return self.s3c.get_weights()

        assert False

    def redo_everything(self):

        #we don't call redo_everything on s3c because this would reset its weights
        #however, calling redo_everything on the dbm just resets its negative chain
        self.dbm.redo_everything()

        if self.sub_batch:
             self.grads = {}
             for param in self.get_params():
                 self.grads[param] = sharedX(np.zeros(param.get_value().shape))


        if self.use_diagonal_natural_gradient:
            self.params_to_means = {}
            self.params_to_M2s = {}
            self.n_var_samples = sharedX(0.0)

            for param in self.get_params():
                self.params_to_means[param] = \
                        sharedX(np.zeros(param.get_value().shape))
                self.params_to_M2s[param] = \
                        sharedX(np.zeros(param.get_value().shape))

        if self.momentum_saturation_example is not None:
            assert not self.use_diagonal_natural_gradient
            self.params_to_incs = {}

            for param in self.get_params():
                self.params_to_incs[param] = sharedX(np.zeros(param.get_value().shape), name = param.name+'_inc')

            self.momentum = sharedX(self.init_momentum, name='momentum')

        if self.non_s3c_lr_saturation_example is not None:
            self.non_s3c_lr = sharedX(self.init_non_s3c_lr, name = 'non_s3c_lr')

        self.test_batch_size = 2

        self.redo_theano()


    def set_dtype(self, dtype):

        assert self.s3c.bias_hid is self.dbm.bias_vis
        super(PDDBM, self).set_dtype(dtype)
        self.s3c.bias_hid = self.dbm.bias_vis

    def get_monitoring_channels(self, V, Y = None):

        if Y is None:
            assert self.dbm.num_classes == 0
        if self.dbm.num_classes == 0:
            Y = None

        try:
            self.compile_mode()

            self.s3c.set_monitoring_channel_prefix('s3c_')

            rval = self.s3c.get_monitoring_channels(V)

            # % of DBM W[0] weights that are negative
            negs = T.sum(T.cast(T.lt(self.dbm.W[0],0.),'float32'))
            total = np.cast['float32'](self.dbm.rbms[0].nvis * self.dbm.rbms[0].nhid)
            negprop = negs /total
            rval['dbm_W[0]_negprop'] = negprop


            rval['shrink'] = self.shrink

            if self.monitor_neg_chain_marginals:
                assert self.dbm.negative_chains > 0
                V_mean_samples = self.s3c.random_design_matrix(batch_size = self.dbm.negative_chains,
                        H_sample = self.dbm.V_chains,
                        S_sample = self.s3c.mu.dimshuffle('x',0), full_sample = False)
                V_mean = V_mean_samples.mean(axis=0)
                rval['marginal_V_mean_min'] = V_mean.min()
                rval['marginal_V_mean_mean'] = V_mean.mean()
                rval['marginal_V_mean_max'] = V_mean.max()



            #DBM negative chain
            H_chain = self.dbm.V_chains.mean(axis=0)
            rval['neg_chain_h_min'] = full_min(H_chain)
            rval['neg_chain_h_mean'] = H_chain.mean()
            rval['neg_chain_h_max'] = full_max(H_chain)
            if self.dbm.Y_chains is not None:
                Y_chain = self.dbm.Y_chains.mean(axis=0)
                rval['neg_chain_y_min'] = full_min(Y_chain)
                rval['neg_chain_y_mean'] = Y_chain.mean()
                rval['neg_chain_y_max'] = full_max(Y_chain)
            G_chains = self.dbm.H_chains
            for i, G_chain in enumerate(G_chains):
                G_chain = G_chain.mean(axis=0)
                rval['neg_chain_g[%d]_min'%i] = full_min(G_chain)
                rval['neg_chain_g[%d]_mean'%i] = G_chain.mean()
                rval['neg_chain_g[%d]_max'%i] = full_max(G_chain)
            rb, rby = self.dbm.rao_blackwellize( self.dbm.V_chains, G_chains, self.dbm.Y_chains)
            for i, rbg in enumerate(rb):
                rbg = rbg.mean(axis=0)
                rval['neg_chain_rbg[%d]_min'%i] = full_min(rbg)
                rval['neg_chain_rbg[%d]_mean'%i] = rbg.mean()
                rval['neg_chain_rbg[%d]_max'%i] = full_max(rbg)
            if rby is not None:
                rby = rby.mean(axis=0)
                rval['neg_chain_rby_min'] = full_min(rby)
                rval['neg_chain_rby_mean'] = rby.mean()
                rval['neg_chain_rby_max'] = full_max(rby)

            from_inference_procedure = self.inference_procedure.get_monitoring_channels(V,Y)

            rval.update(from_inference_procedure)


            self.dbm.set_monitoring_channel_prefix('dbm_')

            from_dbm = self.dbm.get_monitoring_channels(V)

            #remove the s3c bias_hid channels from the DBM's output
            keys_to_del = []
            for key in from_dbm:
                if key.startswith('dbm_bias_hid_'):
                    keys_to_del.append(key)
            for key in keys_to_del:
                del from_dbm[key]

            rval.update(from_dbm)

            if self.use_diagonal_natural_gradient:
                for param in self.get_params():
                    name = 'grad_var_'+param.name

                    params_to_variances = self.get_params_to_variances()

                    var = params_to_variances[param]

                    rval[name+'_min'] = var.min()
                    rval[name+'_mean'] = var.mean()
                    rval[name+'_max'] = var.max()

            if self.momentum_saturation_example is not None:
                rval['momentum'] = self.momentum
            if self.non_s3c_lr_saturation_example is not None:
                rval['non_s3c_lr'] = self.non_s3c_lr

        finally:
            self.deploy_mode()

        return rval

    def get_output_space(self):
        return self.dbm.get_output_space()

    def compile_mode(self):
        """ If any shared variables need to have batch-size dependent sizes,
        sets them all to the sizes used for interactive debugging during graph construction """
        pass

    def deploy_mode(self):
        """ If any shared variables need to have batch-size dependent sizes, sets them all to their runtime sizes """
        pass

    def get_params(self):

        params = set([])

        if not self.freeze_s3c_params:
            params = params.union(set(self.s3c.get_params()))

        if not self.freeze_dbm_params:
            params = params.union(set(self.dbm.get_params()))
        else:
            assert False #temporary debugging assert

        assert self.dbm.bias_hid[0] in params

        return list(params)

    def make_reset_grad_func(self):
        """
        For use with the sub_batch feature only
        Resets the gradient to the data-independent gradient (ie, negative phase, regularization)
        One can then accumulate the positive phase gradient in sub-batches
        """

        assert self.sub_batch

        assert self.g_penalties is None
        assert self.h_penalty == 0.0
        assert self.dbm_weight_decay is None
        assert self.dbm_l1_weight_decay is None
        assert not self.use_cd

        params_to_approx_grads = self.dbm.get_neg_phase_grads()

        updates = {}

        for param in self.grads:
            if param in params_to_approx_grads:
                updates[self.grads[param]] = params_to_approx_grads[param]
            else:
                updates[self.grads[param]] = T.zeros_like(param)

        sampling_updates = self.dbm.get_sampling_updates()

        for key in sampling_updates:
            assert key not in updates
            updates[key] = sampling_updates[key]

        f = function([], updates = updates)

        return f

    def positive_phase_obj(self, V, hidden_obs):
        """ returns both the objective AND things that should be considered constant
            in order to avoid propagating through inference """

        #make a restricted dictionary containing only vars s3c knows about
        restricted_obs = {}
        for key in hidden_obs:
            if key != 'G_hat':
                restricted_obs[key] = hidden_obs[key]


        #request s3c sufficient statistics
        needed_stats = \
         S3C.expected_log_prob_v_given_hs_needed_stats().union(\
         S3C.expected_log_prob_s_given_h_needed_stats())
        stats = SufficientStatistics.from_observations(needed_stats = needed_stats,
                V = V, **restricted_obs)

        #don't backpropagate through inference
        obs_set = set(hidden_obs.values())
        stats_set = set(stats.d.values())
        constants = flatten(obs_set.union(stats_set))

        G_hat = hidden_obs['G_hat']
        for i, G in enumerate(G_hat):
            G.name = 'final_G_hat[%d]' % (i,)
        H_hat = hidden_obs['H_hat']
        H_hat.name = 'final_H_hat'
        S_hat = hidden_obs['S_hat']
        S_hat.name = 'final_S_hat'

        expected_log_prob_v_given_hs = self.s3c.expected_log_prob_v_given_hs(stats, \
                H_hat = H_hat, S_hat = S_hat)
        assert len(expected_log_prob_v_given_hs.type.broadcastable) == 0


        expected_log_prob_s_given_h  = self.s3c.expected_log_prob_s_given_h(stats)
        assert len(expected_log_prob_s_given_h.type.broadcastable) == 0


        expected_dbm_energy = self.dbm.expected_energy( V_hat = H_hat, H_hat = G_hat )
        assert len(expected_dbm_energy.type.broadcastable) == 0

        #note: this is not the complete tractable part of the objective
        #the objective also includes the entropy of Q, but we drop that since it is
        #not a function of the parameters and we're not able to compute the true
        #value of the objective function anyway
        obj = expected_log_prob_v_given_hs + \
                        expected_log_prob_s_given_h  - \
                        expected_dbm_energy

        assert len(obj.type.broadcastable) == 0

        return obj, constants

    def make_grad_step_func(self):

        learning_updates = self.get_param_updates(self.grads)
        self.censor_updates(learning_updates)



        print "compiling function..."
        t1 = time.time()
        rval = function([], updates = learning_updates)
        t2 = time.time()
        print "... compilation took "+str(t2-t1)+" seconds"

        return rval



    def make_learn_func(self, V, Y):
        """
        V: a symbolic design matrix
        Y: None or a symbolic label matrix, one label per row, one-hot encoding
        """

        assert self.inference_procedure is not None

        #obtain the results of variational inference
        hidden_obs = self.inference_procedure.hidden_obs

        #make a restricted dictionary containing only vars s3c knows about
        restricted_obs = {}
        for key in hidden_obs:
            if key not in ['G_hat','Y_hat']:
                restricted_obs[key] = hidden_obs[key]


        #request s3c sufficient statistics
        needed_stats = \
         S3C.expected_log_prob_v_given_hs_needed_stats().union(\
         S3C.expected_log_prob_s_given_h_needed_stats())
        stats = SufficientStatistics.from_observations(needed_stats = needed_stats,
                V = V, **restricted_obs)

        #don't backpropagate through inference
        no_nones = {}
        for key in hidden_obs:
            if key == 'Y_hat' and hidden_obs[key] is None:
                continue
            no_nones[key] = hidden_obs[key]
        obs_set = set(flatten(no_nones.values()))
        stats_set = set(stats.d.values())
        constants = flatten(obs_set.union(stats_set))

        G_hat = hidden_obs['G_hat']
        for i, G in enumerate(G_hat):
            G.name = 'final_G_hat[%d]' % (i,)
        H_hat = hidden_obs['H_hat']
        H_hat.name = 'final_H_hat'
        S_hat = hidden_obs['S_hat']
        S_hat.name = 'final_S_hat'

        assert H_hat in constants
        for G in G_hat:
            assert G in constants
        assert S_hat in constants

        expected_log_prob_v_given_hs = self.s3c.expected_log_prob_v_given_hs(stats, \
                H_hat = H_hat, S_hat = S_hat)
        assert len(expected_log_prob_v_given_hs.type.broadcastable) == 0


        expected_log_prob_s_given_h  = self.s3c.expected_log_prob_s_given_h(stats)
        assert len(expected_log_prob_s_given_h.type.broadcastable) == 0


        expected_dbm_energy = self.dbm.expected_energy( V_hat = H_hat, H_hat = G_hat, Y_hat = Y )
        assert len(expected_dbm_energy.type.broadcastable) == 0

        test = T.grad(expected_dbm_energy, self.dbm.W[0], consider_constant = constants)

        #note: this is not the complete tractable part of the objective
        #the objective also includes the entropy of Q, but we drop that since it is
        #not a function of the parameters and we're not able to compute the true
        #value of the objective function anyway
        tractable_obj = expected_log_prob_v_given_hs + \
                        expected_log_prob_s_given_h  - \
                        expected_dbm_energy

        assert len(tractable_obj.type.broadcastable) == 0


        if self.dbm_weight_decay:

            for i, t in enumerate(zip(self.dbm_weight_decay, self.dbm.W)):

                coeff, W = t

                coeff = as_floatX(float(coeff))
                coeff = T.as_tensor_variable(coeff)
                coeff.name = 'dbm_weight_decay_coeff_'+str(i)

                tractable_obj = tractable_obj - coeff * T.mean(T.sqr(W))

        if self.dbm_l1_weight_decay:

            for i, t in enumerate(zip(self.dbm_l1_weight_decay, self.dbm.W)):

                coeff, W = t

                coeff = as_floatX(float(coeff))
                coeff = T.as_tensor_variable(coeff)
                coeff.name = 'dbm_l1_weight_decay_coeff_'+str(i)

                tractable_obj = tractable_obj - coeff * T.sum(abs(W))

        if self.h_penalty != 0.0:
            next_h = self.inference_procedure.infer_H_hat(V = V,
                H_hat = H_hat, S_hat = S_hat, G1_hat = G_hat[0])

            #err = next_h.mean(axis=0) - self.h_target

            #abs_err = abs(err)

            penalty = T.sum( T.nnet.binary_crossentropy( target = self.h_target,
                output = next_h.mean(axis=0)) )

            tractable_obj =  tractable_obj - self.h_penalty * penalty

        if self.g_penalties is not None:
            for i in xrange(len(self.dbm.bias_hid)):
                G = self.inference_procedure.infer_G_hat(H_hat = H_hat, G_hat = G_hat, idx = i)

                g = T.mean(G,axis=0)

                #err = g - self.g_targets[i]

                #abs_err = abs(err)

                #penalty = T.mean(abs_err)

                penalty = T.sum( T.nnet.binary_crossentropy( target = self.g_targets[i], output = g))

                tractable_obj = tractable_obj - self.g_penalties[i] * penalty

        if self.s3c_l1_weight_decay != 0.0:

            tractable_obj = tractable_obj - self.s3c_l1_weight_decay * T.mean(abs(self.s3c.W))

        if self.B_penalty != 0.0:
            tractable_obj = tractable_obj - T.mean(self.s3c.B) * self.B_penalty

        if self.alpha_penalty != 0.0:
            tractable_obj = tractable_obj - T.mean(self.s3c.alpha) * self.alpha_penalty


        if self.recons_penalty is not None:
            tractable_obj = tractable_obj - self.recons_penalty * self.simple_recons_error(V, G_hat)

        assert len(tractable_obj.type.broadcastable) == 0
        assert tractable_obj.type.dtype == config.floatX

        #take the gradient of the tractable part
        params = self.get_params()
        grads = T.grad(tractable_obj, params, consider_constant = constants, disconnected_inputs = 'warn')

        #put gradients into convenient dictionary
        params_to_grads = {}
        for param, grad in zip(params, grads):
            params_to_grads[param] = grad

        #make function for online estimate of variance of grad
        #this is kind of a hack, since I install the function rather
        #than returning it. should clean this up

        if self.use_diagonal_natural_gradient:
            new_n = self.n_var_samples + as_floatX(1.)

            var_updates = { self.n_var_samples : new_n }

            for param in params:
                grad = params_to_grads[param]
                mean = self.params_to_means[param]
                M2 = self.params_to_M2s[param]

                delta = grad - mean
                new_mean = mean + delta / new_n

                var_updates[M2] = M2 + delta * (grad - new_mean)
                var_updates[mean] = new_mean

            self.update_variances = function([V], updates = var_updates)

        #end hacky part


        #add the approximate gradients
        if self.use_cd:
            params_to_approx_grads = self.dbm.get_cd_neg_phase_grads(V = H_hat, H_hat = G_hat, Y = Y)
        else:
            params_to_approx_grads = self.dbm.get_neg_phase_grads()

        for param in params_to_approx_grads:
            if param in params_to_grads:
                params_to_grads[param] = params_to_grads[param] + params_to_approx_grads[param]
                params_to_grads[param].name = param.name + '_final_approx_grad'

        if self.use_diagonal_natural_gradient:

            params_to_variances = self.get_params_to_variances()

            for param in set(self.dbm.W).union(self.dbm.bias_hid):

                grad = params_to_grads[param]
                var = params_to_variances[param]
                safe_var = var + as_floatX(.5)
                scaled_grad = grad / safe_var
                params_to_grads[param] = scaled_grad

        assert self.dbm.bias_hid[0] in params_to_grads
        learning_updates = self.get_param_updates(params_to_grads)
        assert self.dbm.bias_hid[0] in learning_updates

        if self.use_cd:
            sampling_updates = {}
        else:
            sampling_updates = self.dbm.get_sampling_updates()

        for key in sampling_updates:
            learning_updates[key] = sampling_updates[key]

        self.censor_updates(learning_updates)


        #print 'learning updates contains: '
        #for key in learning_updates:
        #    print '\t',key
        #print min_informative_str(learning_updates[self.dbm.bias_hid[0]])
        #assert False

        inputs = [ V ]

        if Y is not None:
            inputs.append(Y)


        print "compiling PD-DBM learn function..."
        t1 = time.time()
        rval = function(inputs, updates = learning_updates)
        t2 = time.time()
        print "... compilation took "+str(t2-t1)+" seconds"
        print "graph size: ",len(rval.maker.env.toposort())

        return rval


    def simple_recons_error(self, V, G_hat):
        """
            makes a single downward meanfield pass from the deepest layer
            to estimate a reconstruction
            returns the mean squared error of that reconstruction
            NOTE: alpha has no effect on this. we might want to do
            E[ error(V,recons) ] rather than error(V,E[recons]) so
            that alpha gets encouraged to be small
        """

        assert len(G_hat) == 1
        H = T.nnet.sigmoid(
                T.dot(G_hat[0], self.dbm.W[0].T) + self.dbm.bias_vis)
        HS = H * self.s3c.mu
        recons = T.dot(HS, self.s3c.W.T)

        return T.mean(T.sqr(recons - V))

    def get_param_updates(self, params_to_grads):

        warnings.warn("TODO: get_param_updates does not use geodesics for now")

        rval = {}

        learning_rate = {}

        for param in params_to_grads:
            if param is self.s3c.B_driver:
                learning_rate[param] = as_floatX(self.learning_rate * self.B_learning_rate_scale)
            elif param is self.s3c.alpha:
                learning_rate[param] = as_floatX(self.learning_rate * self.alpha_learning_rate_scale)
            elif param is self.s3c.W:
                learning_rate[param] = as_floatX(self.learning_rate * self.s3c_W_learning_rate_scale)
            elif param is self.s3c.mu:
                learning_rate[param] = as_floatX(self.learning_rate * self.s3c_mu_learning_rate_scale)
            elif param not in self.s3c.get_params():
                if self.non_s3c_lr_saturation_example is not None:
                    learning_rate[param] = self.non_s3c_lr
                else:
                    learning_rate[param] = as_floatX(self.learning_rate)
            else:
                learning_rate[param] = as_floatX(self.learning_rate)

        self.shrink = sharedX(1.0)

        for param in learning_rate:
            learning_rate[param] = self.shrink * learning_rate[param]

        if self.momentum_saturation_example is not None:
            for key in params_to_grads:
                inc = self.params_to_incs[key]
                rval[inc] = self.momentum * inc + learning_rate[key] * params_to_grads[key]
                rval[key] = key + rval[inc]
        else:
            for key in params_to_grads:
                rval[key] = key + learning_rate[key] * params_to_grads[key]


        for param in self.get_params():
            assert param in params_to_grads
            assert param in rval

        assert self.dbm.bias_hid[0] in rval

        return rval

    def censor_updates(self, updates):

        if self.freeze_s3c_params:
            for param in self.s3c.get_params():
                assert param not in updates or param is self.dbm.bias_vis

        if self.freeze_dbm_params:
            for param in self.dbm.get_params():
                if param in updates and param is not self.s3c.bias_hid:
                    assert hasattr(param,'name')
                    name = 'anon'
                    if param.name is not None:
                        name = param.name
                    raise AssertionError("DBM parameters are frozen but you're trying to update DBM parameter "+name)

        self.s3c.censor_updates(updates)
        self.dbm.censor_updates(updates)

    def random_design_matrix(self, batch_size, theano_rng):

        if not hasattr(self,'p'):
            self.make_pseudoparams()

        H_sample = self.dbm.random_design_matrix(batch_size, theano_rng)

        V_sample = self.s3c.random_design_matrix(batch_size, theano_rng, H_sample = H_sample)

        return V_sample


    def make_pseudoparams(self):
        self.s3c.make_pseudoparams()


    def get_params_to_variances(self):

        rval = {}

        for param in self.get_params():
            M2 = self.params_to_M2s[param]
            rval[param] = M2 / (self.n_var_samples - as_floatX(1.))

        return rval

    def redo_theano(self):

        self.s3c.reset_censorship_cache()

        try:
            self.compile_mode()

            init_names = dir(self)

            self.make_pseudoparams()

            if self.inference_procedure is not None:
                self.inference_procedure.redo_theano()

            X = T.matrix(name='V')
            X.tag.test_value = np.cast[config.floatX](self.rng.randn(self.test_batch_size,self.nvis))

            if self.dbm.num_classes > 0:
                Y = T.matrix(name='Y')
            else:
                Y = None

            if self.use_diagonal_natural_gradient:
                updates = { self.n_var_samples : as_floatX(0.0) }

                for param in self.get_params():
                    updates[self.params_to_means[param]] = \
                            0. * self.params_to_means[param]
                    updates[self.params_to_M2s[param]] = \
                            0. * self.params_to_M2s[param]
                self.reset_variances = function([],updates= updates)

            self.s3c.e_step.register_model(self.s3c)

            if self.sub_batch:
                if Y is not None:
                    raise NotImplementedError("sub_batch mode does not support labels yet")
                self.reset_grad_func = self.make_reset_grad_func()
                self.accum_pos_phase_grad_func = self.make_accum_pos_phase_grad_func(X)
                self.grad_step_func = self.make_grad_step_func()
            else:
                if self.inference_procedure is not None:
                    self.learn_func = self.make_learn_func(X,Y)

            final_names = dir(self)

            self.register_names_to_del([name for name in final_names if name not in init_names])
        finally:
            self.deploy_mode()
        #end try block
    #end redo_theano

    def learn(self, dataset, batch_size):


        if self.bayes_B:
            self.bayes_B = False

            var = dataset.X.var(axis=0)

            assert not self.s3c.tied_B
            self.s3c.B_driver.set_value( 1. / (var + .01) )

        if self.exhaustive_iteration:
            def make_iterator():
                self.iterator = dataset.iterator(
                        mode = 'sequential',
                        batch_size = batch_size,
                        targets = self.dbm.num_classes > 0)

            if self.iterator is None:
                self.batch_size = batch_size
                self.dataset = dataset
                self.register_names_to_del(['dataset','iterator'])
                make_iterator()
            else:
                assert dataset is self.dataset
                assert batch_size == self.batch_size
            if self.dbm.num_classes > 0:
                try:
                    X, Y = self.iterator.next()
                except StopIteration:
                    print 'Finished a dataset-epoch'
                    make_iterator()
                    X, Y = self.iterator.next()
            else:
                Y = None
                try:
                    X = self.iterator.next()
                except StopIteration:
                    print 'Finished a dataset-epoch'
                    make_iterator()
                    X = self.iterator.next()
        else:
            if self.dbm.num_classes > 0:
                raise NotImplementedError("Random iteration doesn't support using class labels yet")
            X = dataset.get_batch_design(batch_size)
            Y = None

        self.learn_mini_batch(X,Y)

    def learn_mini_batch(self, X, Y = None):

        assert (Y is None) == (self.dbm.num_classes == 0)

        self.shrink.set_value( np.cast['float32']( \
                max(self.min_shrink,
                    1. / (1. + self.lr_shrink_example_scale * float( \
                            max(0,self.monitor.get_examples_seen() - float(self.shrink_start)))))))

        assert self.s3c is self.inference_procedure.s3c_e_step.model
        if self.momentum_saturation_example is not None:
            alpha = float(self.monitor.get_examples_seen()) / float(self.momentum_saturation_example)
            alpha = min( alpha, 1.0)
            self.momentum.set_value(np.cast[config.floatX](
                (1.-alpha) * self.init_momentum + alpha * self.final_momentum))
        if self.non_s3c_lr_saturation_example is not None:
            alpha = (float(self.monitor.get_examples_seen()) - float(self.non_s3c_lr_start)) / (float(self.non_s3c_lr_saturation_example) - float(self.non_s3c_lr_start))
            alpha = max( alpha, 0.0)
            alpha = min( alpha, 1.0)
            self.non_s3c_lr.set_value(np.cast[config.floatX](
                (1.-alpha) * self.init_non_s3c_lr + alpha * self.final_non_s3c_lr))



        if self.use_diagonal_natural_gradient:
            if self.dbm.num_classes > 0:
                raise NotImplementedError()
            self.reset_variances()
            for i in xrange(X.shape[0]):
                self.update_variances(X[i:i+1,:])

        if self.sub_batch:
            self.reset_grad_func()
            for i in xrange(X.shape[0]):
                self.accum_pos_phase_grad_func(X[i:i+1,:])
        else:
            if Y is None:
                self.inference_procedure.update_var_params(X)
                self.learn_func(X)
            else:
                self.inference_procedure.update_var_params(X,Y)
                self.learn_func(X,Y)
        if self.monitor._examples_seen % self.print_interval == 0:
            print ""
            print "S3C:"
            self.s3c.print_status()
            print "DBM:"
            self.dbm.print_status()

    def get_weights_format(self):
        return self.s3c.get_weights_format()

class MonitorPrereq:

    def __init__(self, ip):
        self.ip = ip

    def __call__(self, X, Y = None):
        return self.ip.update_var_params(X,Y)

class InferenceProcedure(Model):
    """

    Variational inference

    This is only a Model in order to automatically get access to the Model's
    pickle mangling abilities.

    """

    def __init__(self,
                        clip_reflections = False,
                       rho = 0.5,
                       list_update_new_coeff = 1e-2,
                       monitor_kl_fail = False):
        """Parameters
        --------------
        schedule:

        clip_reflections, rho : if clip_reflections is true, the update to Mu1[i,j] is
            bounded on one side by - rho * Mu1[i,j] and unbounded on the other side
        """

        super(InferenceProcedure, self).__init__()

        self.list_update_new_coeff = list_update_new_coeff
        self.clip_reflections = clip_reflections
        self.rho = as_floatX(rho)

        self.model = None

        self.monitor_prereq = MonitorPrereq(self)
        self.monitor_kl_fail = monitor_kl_fail


    def get_monitoring_channels(self, V, Y = None):

        assert (Y is None) == (self.model.dbm.num_classes == 0)

        rval = {}

        rval['trunc_KL_init'] = self.kl_init
        rval['trunc_KL_final'] = self.kl_final
        rval['time_per_ex'] = self.time

        if self.monitor_kl_fail:
            rval['kl_fail'] = (self.kl_fail_object.kl_fail, (self.kl_fail_object,))

        final_vals = self.hidden_obs

        S_hat = final_vals['S_hat']
        H_hat = final_vals['H_hat']
        h = T.mean(H_hat, axis=0)

        rval['h_min'] = full_min(h)
        rval['h_mean'] = T.mean(h)
        rval['h_max'] = full_max(h)

        Gs = final_vals['G_hat']

        for i, G in enumerate(Gs):

            g = T.mean(G,axis=0)

            rval['g[%d]_min'%(i,)] = full_min(g)
            rval['g[%d]_mean'%(i,)] = T.mean(g)
            rval['g[%d]_max'%(i,)] = full_max(g)


        #norm of gradient with respect to variational params
        grad_norm_sq = np.cast[config.floatX](0.)
        kl = self.truncated_KL(V, final_vals, Y).mean()
        for var_param in set([ S_hat, H_hat]).union(Gs):
            grad = T.grad(kl,var_param)
            grad_norm_sq = grad_norm_sq + T.sum(T.sqr(grad))
        grad_norm = T.sqrt(grad_norm_sq)
        rval['var_param_grad_norm'] = grad_norm

        if self.model.monitor_ranges:
            S_hat = final_vals['S_hat']
            HS = H_hat * S_hat

            hs_max = T.max(HS,axis=0)
            hs_min = T.min(HS,axis=0)

            hs_range = hs_max - hs_min

            rval['hs_range_min'] = T.min(hs_range)
            rval['hs_range_mean'] = T.mean(hs_range)
            rval['hs_range_max'] = T.max(hs_range)

            h_max = T.max(H_hat,axis=0)
            h_min = T.min(H_hat,axis=0)

            h_range = h_max - h_min

            rval['h_range_min'] = T.min(h_range)
            rval['h_range_mean'] = T.mean(h_range)
            rval['h_range_max'] = T.max(h_range)

            for i, G in enumerate(Gs):

                g_max = T.max(G,axis=0)
                g_min = T.min(G,axis=0)

                g_range = g_max - g_min

                g_name = 'g[%d]' % (i,)

                rval[g_name+'_range_min'] = T.min(g_range)
                rval[g_name+'_range_mean'] = T.mean(g_range)
                rval[g_name+'_range_max'] = T.max(g_range)


        #if self.model.recons_penalty is not None:
        rval['simple_recons_error'] = self.model.simple_recons_error(V,Gs)


        for key in rval:
            #kl_fail already has a prereq, the kl fail test
            #give the other channels a prereq of inference
            if not isinstance(rval[key],tuple):
                rval[key] = (rval[key], (self.monitor_prereq,))

        return rval

    def register_model(self, model):
        self.model = model

        self.s3c_e_step = self.model.s3c.e_step
        self.s3c_e_step.register_model(model.s3c)

        self.s3c_e_step.clip_reflections = self.clip_reflections
        self.s3c_e_step.rho = self.rho

        self.dbm_ip = self.model.dbm.inference_procedure


        codes = ['s','h','y']
        for i in xrange(len(self.model.dbm.rbms)):
            codes.append(('g',i))

        self.new_coeff_lists = {}
        self.default_new_coeff = {}
        self.tols = {}
        for code in codes:
            self.new_coeff_lists[code] = []
            self.default_new_coeff[code] = 1.
            self.tols[code] = 1e-3


    def dbm_observations(self, obs):

        rval = {}
        rval['H_hat'] = obs['G_hat']
        rval['V_hat'] = obs['H_hat']

        return rval

    def truncated_KL(self, V, obs, Y = None):
        """ KL divergence between variational and true posterior, dropping terms that don't
            depend on the variational parameters """

        for G_hat in obs['G_hat']:
            for Gv in get_debug_values(G_hat):
                assert Gv.min() >= 0.0
                assert Gv.max() <= 1.0

        s3c_truncated_KL = self.s3c_e_step.truncated_KL(V, Y = None, obs = obs)
        assert len(s3c_truncated_KL.type.broadcastable) == 1

        dbm_obs = self.dbm_observations(obs)

        dbm_truncated_KL = self.dbm_ip.truncated_KL(V = obs['H_hat'], Y = Y, obs = dbm_obs, no_v_bias = True)
        assert len(dbm_truncated_KL.type.broadcastable) == 1

        for s3c_kl_val, dbm_kl_val in get_debug_values(s3c_truncated_KL, dbm_truncated_KL):
            debug_assert( not np.any(np.isnan(s3c_kl_val)))
            debug_assert( not np.any(np.isnan(dbm_kl_val)))

        rval = s3c_truncated_KL + dbm_truncated_KL

        return rval

    def infer_H_hat(self, V, H_hat, S_hat, G1_hat):
        """
            G1_hat: variational parameters for the g layer closest to h
                    here we use the designation from the math rather than from
                    the list, where it is G_hat[0]
        """

        s3c_presigmoid = self.s3c_e_step.infer_H_hat_presigmoid(V, H_hat, S_hat)

        W = self.model.dbm.W[0]

        assert self.model.s3c.bias_hid is self.model.dbm.bias_vis

        top_down = T.dot(G1_hat, W.T)

        presigmoid = s3c_presigmoid + top_down

        H = T.nnet.sigmoid(presigmoid)

        return H


    def infer_G_hat(self, H_hat, G_hat, idx, Y_hat = None):

        assert (Y_hat is None) == (self.model.dbm.num_classes == 0)

        number = idx
        dbm_ip = self.model.dbm.inference_procedure

        b = self.model.dbm.bias_hid[number]

        W = self.model.dbm.W

        W_below = W[number]

        if number == 0:
            H_hat_below = H_hat
        else:
            H_hat_below = G_hat[number - 1]

        num_g = self.model.num_g

        if number == num_g - 1:
            if Y_hat is None:
                return dbm_ip.infer_H_hat_one_sided(other_H_hat = H_hat_below, W = W_below, b = b)
            else:
                for Y_hat_v in get_debug_values(Y_hat):
                    assert Y_hat_v.shape[1] == self.model.dbm.num_classes
                    assert self.model.dbm.num_classes == 10 #temporary debugging assert, can remove
                return dbm_ip.infer_H_hat_two_sided(H_hat_below = H_hat_below, W_below = W_below, b = b,
                        H_hat_above = Y_hat, W_above = self.model.dbm.W_class)
        else:
            H_hat_above = G_hat[number + 1]
            W_above = W[number+1]
            return dbm_ip.infer_H_hat_two_sided(H_hat_below = H_hat_below, H_hat_above = H_hat_above,
                                   W_below = W_below, W_above = W_above,
                                   b = b)

    def infer_var_s1_hat(self):
        return self.s3c_e_step.infer_var_s1_hat()


    def infer_var_s0_hat(self):
        return self.s3c_e_step.infer_var_s0_hat()

    def redo_theano(self):

        init_names = dir(self)

        if self.monitor_kl_fail:
            self.kl_fail_object = BatchGradientInferenceMonitorHack(self.model)

        assert self.model is not None

        batch_size = self.model.test_batch_size


        num_layers = len(self.model.dbm.rbms)

        self.hidden_obs = {}
        N_H = self.model.s3c.nhid
        #don't use zeros here, that's not a valid precision and it will screw up
        #the interactive debugger with NaNs
        self.hidden_obs['var_s0_hat'] = sharedX(np.ones((N_H,)),'var_s0_hat')
        self.hidden_obs['var_s1_hat'] = sharedX(np.ones((N_H,)),'var_s1_hat')
        self.hidden_obs['H_hat'] = sharedX(np.zeros((batch_size,N_H)),'H_hat')
        self.hidden_obs['S_hat'] = sharedX(np.zeros((batch_size,N_H)),'S_hat')
        self.kl_init = sharedX(0.0,'kl_init')
        self.kl_final = sharedX(0.0,'kl_final')
        self.time = sharedX(0.0, 'time_per_ex')
        G_hat = []
        layer_sizes = [ rbm.nhid for rbm in self.model.dbm.rbms ]
        for i, layer_size in enumerate(layer_sizes):
            G_hat.append( sharedX(np.zeros((batch_size, layer_size)),'G_hat[%d]' % i))
        self.hidden_obs['G_hat'] = G_hat

        alpha = self.model.s3c.alpha
        s3c_e_step = self.s3c_e_step
        dbm_ip = self.dbm_ip

        var_s0_hat = 1. / alpha
        var_s1_hat = s3c_e_step.infer_var_s1_hat()

        V = T.matrix('V')
        if config.compute_test_value != 'off':
            V.tag.test_value = np.cast[config.floatX](np.zeros((batch_size, self.model.s3c.nvis)))

        num_classes = self.model.dbm.num_classes
        if num_classes > 0:
            self.hidden_obs['Y_hat'] = sharedX(np.zeros((batch_size,num_classes)))
            self.staged_obs['Y_hat'] = sharedX(np.zeros((batch_size,num_classes)))
            init_val = self.dbm.inference_procedure.init_Y_hat()
            self.initialize_Y_hat = function([],
                    updates = { self.hidden_obs['Y_hat'] : init_val })
        else:
            self.hidden_obs['Y_hat'] = None

        init_H_hat = self.s3c_e_step.init_H_hat(V)
        init_S_hat = self.s3c_e_step.init_S_hat(V)


        #endpoints holds undamped updates to variational parameters,
        #to make it cheaper to try out several different damping amounts
        #since we only update one variable at a time, only one element
        #of this dictionary ever makes sense at a time
        self.endpoints = {}
        #staged holds damped updates. only one entry ever makes sense
        #at any given time. this dictionary exists so we don't need to
        #compute the damped update twice. once we find an acceptable
        #damped update, we copy it back to hidden_obs
        self.staged = {}

        def clone_dict(target):
            assert 'Y_hat' in self.hidden_obs
            for key in self.hidden_obs:
                if self.hidden_obs[key] is None:
                    target[key] = None
                elif isinstance(self.hidden_obs[key],list):
                    target[key] = tuple([
                        sharedX(var.get_value()) for var in self.hidden_obs[key]])
                else:
                    target[key] = sharedX(self.hidden_obs[key].get_value())

        clone_dict(self.endpoints)
        assert 'Y_hat' in self.endpoints
        clone_dict(self.staged)

        init_dict = { self.hidden_obs['var_s0_hat'] : var_s0_hat,
                  self.hidden_obs['var_s1_hat'] : var_s1_hat,
                  self.hidden_obs['H_hat'] : init_H_hat,
                  self.hidden_obs['S_hat'] : init_S_hat }

        init_G_hat = self.dbm_ip.init_H_hat(init_H_hat)
        for G_hat in [ G_hat, self.staged['G_hat'] ]:
            for var_elem, val_elem in zip(G_hat, init_G_hat):
                init_dict[var_elem] = val_elem

        self.initialize_inference = function([V], updates = init_dict)


        if self.hidden_obs['Y_hat'] is not None:
            raise NotImplementedError()
            """I think I need to make two different KL divergence functions, one for if
            inference is run inferring Y_hat and one for if it is run with Y clamped """
        init_trunc_kl = self.truncated_KL(V, self.hidden_obs, self.hidden_obs['Y_hat']).mean()
        self.compute_init_trunc_kl = function([V],
                init_trunc_kl,
                updates = { self.kl_init : init_trunc_kl }
                )



        #compute_damp_kl is a dictionary mapping codes for hidden layers to
        #functions that
        #put an undamped version of an update in endpoints ("compute")
        #return the kl divergence of a damped version

        #TODO: for now, for simplicity, we do the full computation of
        #all updates every time through the loop. Later, we should
        #optimize it so that commonly re-computed subexpressions, like
        # v^T W, are only computed once and then re-used on each
        # pass through the loop

        self.compute_damp_kl = {}


        codes = ['s','h']
        codes_to_hiddens = { 's' : self.hidden_obs['S_hat'],
                            'h' : self.hidden_obs['H_hat'],
                            'y' : self.hidden_obs['Y_hat']}
        codes_to_endpoints = { 's' : self.endpoints['S_hat'],
                            'h' : self.endpoints['H_hat'],
                            'y' : self.endpoints['Y_hat']}
        codes_to_staged = { 's' : self.staged['S_hat'],
                'h': self.staged['H_hat'],
                'y': self.staged['Y_hat']}

        new_coeff = T.scalar(name = 'new_coeff')

        if config.compute_test_value != 'off':
            new_coeff.tag.test_value = 1.0

        for i in xrange(num_layers):
            codes.append(('g',i))
            codes_to_hiddens[('g',i)] = self.hidden_obs['G_hat'][i]
            codes_to_endpoints[ ('g',i)] = self.endpoints['G_hat'][i]
            codes_to_staged[ ('g',i) ] = self.staged['G_hat'][i]

        H_hat = codes_to_hiddens['h']
        S_hat = codes_to_hiddens['s']
        G1_hat = codes_to_hiddens[('g',0)]
        Y_hat = codes_to_hiddens['y']
        last_G = self.hidden_obs['G_hat'][-1]

        for code in codes:

            #Find the shared variable containing the original value of the variational parameter
            #to be updated
            orig_var = codes_to_hiddens[code]

            #Find the shared variable to which we will write the value of the undamped update
            endpoint_var = codes_to_endpoints[code]

            #Compute the undamped update
            if code == 's':
                endpoint_val = s3c_e_step.infer_S_hat(V, H_hat, S_hat)
                if self.clip_reflections:
                    endpoint_val = reflection_clip(S_hat = S_hat, new_S_hat = endpoint_val, rho = self.rho)
            elif code == 'h':
                endpoint_val = self.infer_H_hat(V = V, H_hat = H_hat, S_hat = S_hat, G1_hat = G1_hat)
            elif code == 'y':
                endpoint_val = dbm_ip.infer_Y_hat( H_hat = last_G)
            else:
                letter, number = code
                assert letter == 'g'
                endpoint_val = self.infer_G_hat( H_hat = H_hat, G_hat = G_hat, idx = number, Y_hat = Y_hat)

            #Compute the damped update
            damped_val = damp(old = orig_var, new = endpoint_val, new_coeff = new_coeff)

            #Find the shared variable containing the final value of the update
            staged_var = codes_to_staged[code]

            #Make a variational parameter dictionary containing the staged version of the
            #parameter we're currently updating and the old version of all other parameters
            active = {}

            for key in self.hidden_obs:
                active[key] = self.hidden_obs[key]

            if code == 's':
                active['S_hat'] = damped_val
            elif code == 'h':
                active['H_hat'] = damped_val
            else:
                assert code[0] == 'g'
                patched_g = [ elem for elem in self.hidden_obs['G_hat']]
                patched_g[code[1]] = damped_val
                active['G_hat'] = patched_g

            #Compute the KL divergence after the update
            if self.hidden_obs['Y_hat'] is not None:
                raise NotImplementedError()
                """I think I need to make two different KL divergence functions, one for if
                inference is run inferring Y_hat and one for if it is run with Y clamped """
            kl = self.truncated_KL(V, active, active['Y_hat']).mean()

            #Write out the undamped update and the damped update
            updates = {}
            updates[endpoint_var] = endpoint_val
            updates[staged_var] = damped_val

            self.compute_damp_kl[code] = function([V, new_coeff], kl, updates = updates )

        self.damp_funcs = {}

        for code in codes:

            #Find the shared variable containing the original value of the variational parameter
            #to be updated
            orig_var = codes_to_hiddens[code]

            #Find the shared variable containing the undamped update
            endpoint_var = codes_to_endpoints[code]

            #Compute the damped update
            damped_val = damp(old = orig_var, new = endpoint_var, new_coeff = new_coeff)

            #Find the shared variable containing the final value of the update
            staged_var = codes_to_staged[code]

            #Make a variational parameter dictionary containing the staged version of the
            #parameter we're currently updating and the old version of all other parameters
            active = {}

            for key in self.hidden_obs:
                active[key] = self.hidden_obs[key]

            if code == 's':
                active['S_hat'] = damped_val
            elif code == 'h':
                active['H_hat'] = damped_val
            else:
                assert code[0] == 'g'
                patched_g = [ elem for elem in self.hidden_obs['G_hat']]
                patched_g[code[1]] = damped_val
                active['G_hat'] = patched_g

            #Compute the KL divergence after the update
            if self.hidden_obs['Y_hat'] is not None:
                raise NotImplementedError()
                """I think I need to make two different KL divergence functions, one for if
                inference is run inferring Y_hat and one for if it is run with Y clamped """
            kl = self.truncated_KL(V, active, active['Y_hat']).mean()

            #Write out the undamped update and the damped update
            updates = {}
            updates[staged_var] = damped_val

            self.damp_funcs[code] = function([V, new_coeff], kl, updates = updates )

        self.lock_funcs = {}

        for code in codes:
            orig_var = codes_to_hiddens[code]
            staged_var = codes_to_staged[code]
            diff = full_max(abs(staged_var-orig_var))
            self.lock_funcs[code] = function([],diff,updates = {orig_var : staged_var } )

        final_names = dir(self)

        self.register_names_to_del([name for name in final_names if name not in init_names])

        for key in self.hidden_obs:
            value = self.hidden_obs[key]
            if isinstance(value,list):
                for elem in value:
                    if not hasattr(elem, 'get_value'):
                        print elem
                        assert False
            else:
                assert value is None or hasattr(value,'get_value')




    def update_var_params(self, V, Y = None):
        """

            V: a numpy matrix of observed values
            Y:
                None corresponds to a model that has no labels
                -1 corresponds to inferring Y in a model that has labels
                    (each G_hat update will become a G_hat-Y_hat-G_hat update)
                a numpy matrix corresponds to clamping Y_hat to that matrix
        """

        t1 = time.time()

        assert Y not in [True,False,0,1] #detect bug where Y gets something that was meant to be return_history
        assert (Y is None) == (self.model.dbm.num_classes == 0)

        infer_labels = Y == -1

        #set H_hat, S_hat and G_hat to their initial values
        #compute the sigma parameters
        #TODO: since sigma depends only on the model parameters,
        #may want to make an option to not compute it if you
        #know the model parameters haven't change, e.g. during
        #feature extraction
        self.initialize_inference(V)

        if infer_labels:
            self.initialize_Y_hat()
        elif Y is not None:
            self.hidden_obs['Y_hat'].set_value(Y)

        num_layers = len(self.model.dbm.rbms)

        h_tol = 1e-3
        s_tol = 1e-3
        g_tol = [1e-3] * num_layers


        updates = ['s','h']
        for i in xrange(num_layers):
            updates.append(('g',i))
        if infer_labels:
            updates.append('y')


        #for key in self.hidden_obs:
        #    try:
        #        if key == 'G_hat':
        #            G_hat = self.hidden_obs['G_hat']
        #            for i in xrange(len(G_hat)):
        #                print 'G_hat[%d]:' %i, G_hat[i].get_value().shape
        #        else:
        #            print key,':',self.hidden_obs[key].get_value().shape
        #    except:
        #        print "couldn't print",key

        trunc_kl = self.compute_init_trunc_kl(V)
        #print 'init_kl:',trunc_kl

        def do_update(update_code, idx, kl_before):
            code = update_code
            compute_damp_kl = self.compute_damp_kl[code]
            damp_func = self.damp_funcs[code]
            lock = self.lock_funcs[code]
            new_coeff_list = self.new_coeff_lists[code]
            tol = self.tols[code]

            #print 'iteration',idx,code

            n = len(new_coeff_list)
            use_default = idx >= n
            if use_default:
                #print '\texceeded end of list, growing it'
                #the topmost layer gets skipped on odd-numbered iterations so we
                #sometimes have a bunch of unused coefficents sitting around for
                #that layer. this also means we need to grow it 2 elements instead
                #of 1
                len_to_grow = idx - n + 1
                for i in xrange(len_to_grow):
                    new_coeff_list.append(self.default_new_coeff[code])


            coeff_before = new_coeff_list[idx]
            #print '\tusing new_coeff of',coeff_before

            coeff = coeff_before
            new_kl = compute_damp_kl(V, coeff)
            #print '\tachieved kl of',new_kl

            gave_up = False
            while new_kl > kl_before:
                if coeff < .01:
                    gave_up = True
                    new_kl = kl_before
                    break
                coeff *= .9
                #print '\tusing new_coeff of',coeff
                new_kl = damp_func(V,coeff)
                #print '\tachieved kl of',new_kl

            max_diff = lock()

            converged = gave_up or max_diff < tol

            #print '\tmax diff was ',max_diff
            #print '\ttol was ',tol
            #if max_diff < tol:
            #    print '\tconvergence criterion met'

            alpha = self.list_update_new_coeff
            new_coeff_list[idx] = alpha * coeff + (1.-alpha) * coeff_before
            #print '\tupdated damp coeff in list to',new_coeff_list[idx]
            if use_default:
                alpha = 1e-3
                self.default_new_coeff[code] = alpha * coeff + (1.-alpha) * coeff_before
                #print '\tupdated default to',self.default_new_coeff[code]

            #TODO: more aggressive version that tries for good damping coefficients
            #      rather than just getting them to go downhill

            return new_kl, converged




        i = 0

        kls = [ trunc_kl ]
        lookback = 10
        lookback_tol = .05

        while True:
            if i % 2 == 0:
                order = updates
            else:
                order = updates[:-1]
                order.reverse()

            all_should_terminate = True

            for update in order:
                trunc_kl, should_terminate = do_update(update, i, trunc_kl)
                all_should_terminate = all_should_terminate and should_terminate


            if all_should_terminate:
                break

            if len(kls) >= lookback and kls[-lookback] < trunc_kl + lookback_tol:
                print "kl converged after",i,"iterations"
                break

            kls.append(trunc_kl)

            i += 1

            if i == 500:
                print "gave up after 500 iterations"
                break

        self.kl_final.set_value(trunc_kl)

        t2 = time.time()

        time_per_ex = (t2-t1)/float(V.shape[0])

        self.time.set_value(time_per_ex)



def get_s3c(pddbm, W_learning_rate_scale = None):
    """ Modifies an s3c object and extracts it from a pddbm """
    rval =  pddbm.s3c
    if rval.m_step is not None:
        if W_learning_rate_scale is not None:
            rval.m_step.W_learning_rate_scale = W_learning_rate_scale
    return rval

def get_dbm(pddbm):
    return pddbm.dbm
