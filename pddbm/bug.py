from galatea.pddbm.pddbm import PDDBM
from galatea.pddbm.pddbm import InferenceProcedure
from pylearn2.models.dbm import DBM
from pylearn2.models.rbm import RBM
from pylearn2.models.s3c import S3C
from pylearn2.models.s3c import Grad_M_Step
from theano.gof.op import get_debug_values, debug_error_message
from theano import config
from pylearn2.utils import make_name, as_floatX
from theano import tensor as T
from theano.printing import min_informative_str

class DebugInferenceProcedure(InferenceProcedure):
    def infer_H_hat(self, V, H_hat, S_hat, G1_hat):
        W = self.model.dbm.W[0]
        top_down = T.dot(G1_hat, W.T)
        H = T.nnet.sigmoid(top_down)
        return H

    def infer(self, V, return_history = False):
        alpha = self.model.s3c.alpha
        s3c_e_step = self.s3c_e_step
        dbm_ip = self.dbm_ip

        var_s0_hat = 1. / alpha
        var_s1_hat = s3c_e_step.var_s1_hat()

        H_hat = s3c_e_step.init_H_hat(V)
        G_hat = dbm_ip.init_H_hat(H_hat)
        S_hat = s3c_e_step.init_S_hat(V)

        H_hat.name = 'init_H_hat'
        S_hat.name = 'init_S_hat'

        def make_dict():

            return {
                    'G_hat' : tuple(G_hat),
                    'H_hat' : H_hat,
                    'S_hat' : S_hat,
                    'var_s0_hat' : var_s0_hat,
                    #'var_s1_hat': var_s1_hat,
                    }

        for i, step in enumerate(self.schedule):
            letter, number = step

            if letter == 'h':
                H_hat = self.infer_H_hat(V = V, H_hat = H_hat, S_hat = S_hat, G1_hat = G_hat[0])
                H_hat.name = 'new_H_hat_step_'+str(i)
            elif letter == 'g':
                b = self.model.dbm.bias_hid[0]
                W_below = self.model.dbm.W[0]
                H_hat_below = H_hat
                G_hat[number] = dbm_ip.infer_H_hat_one_sided(other_H_hat = H_hat_below, W = W_below, b = b)

        return make_dict()


class DebugDBM(DBM):
    def expected_energy(self, V_hat, H_hat):
        V_name = make_name(V_hat, 'anon_V_hat')

        m = V_hat.shape[0]
        m.name = V_name + '.shape[0]'

        exp_vh = T.dot(V_hat.T,H_hat[0]) / m

        v_weights_contrib = T.sum(self.W[0] * exp_vh)

        return v_weights_contrib


class DebugPDDBM(PDDBM):
    def __init__(self,
            s3c,
            dbm,
            learning_rate,
            inference_procedure,
            print_interval = 10000,
            dbm_weight_decay = None):
        super(DebugPDDBM,self).__init__(s3c,dbm,learning_rate,inference_procedure,print_interval,dbm_weight_decay)

    def make_learn_func(self, V):
        """
        V: a symbolic design matrix
        """

        #run variational inference on the train set
        hidden_obs = self.inference_procedure.infer(V)

        #make a restricted dictionary containing only vars s3c knows about
        restricted_obs = {}
        for key in hidden_obs:
            if key != 'G_hat':
                restricted_obs[key] = hidden_obs[key]


        #request s3c sufficient statistics
        needed_stats = \
         S3C.expected_log_prob_v_given_hs_needed_stats().union(\
         S3C.log_likelihood_s_given_h_needed_stats())
         #stats = SufficientStatistics.from_observations(needed_stats = needed_stats,
         #       V = V, **restricted_obs)

        #don't backpropagate through inference
        obs_set = set(hidden_obs.values())
        #stats_set = set(stats.d.values())
        constants = obs_set
        #constants = obs_set.union(stats_set)

        G_hat = hidden_obs['G_hat']
        for i, G in enumerate(G_hat):
            G.name = 'final_G_hat[%d]' % (i,)
        H_hat = hidden_obs['H_hat']
        H_hat.name = 'final_H_hat'
        S_hat = hidden_obs['S_hat']
        S_hat.name = 'final_S_hat'

        assert H_hat in constants
        assert G_hat in constants
        assert S_hat in constants

        #expected_log_prob_v_given_hs = self.s3c.expected_log_prob_v_given_hs(stats, \
        #        H_hat = H_hat, S_hat = S_hat)
        #assert len(expected_log_prob_v_given_hs.type.broadcastable) == 0


        #expected_log_prob_s_given_h  = self.s3c.log_likelihood_s_given_h(stats)
        #assert len(expected_log_prob_s_given_h.type.broadcastable) == 0


        expected_dbm_energy = self.dbm.expected_energy( V_hat = H_hat, H_hat = G_hat )
        assert len(expected_dbm_energy.type.broadcastable) == 0

        #warnings.warn("""need to debug:
        test = T.grad(expected_dbm_energy, self.dbm.W[0], consider_constant = constants)
        print min_informative_str(test)
        assert False
        #""")

        warnings.warn(""" also need to debug
        test = T.grad(expected_dbm_energy, self.dbm.W[0])
        print min_informative_str(test)
        assert False
        """)


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

                coeff = as_floatX(coeff)
                coeff = T.as_tensor_variable(coeff)
                coeff.name = 'dbm_weight_decay_coeff_'+str(i)

                tractable_obj = tractable_obj - coeff * T.mean(T.sqr(W))

        assert len(tractable_obj.type.broadcastable) == 0

        #take the gradient of the tractable part
        params = self.get_params()
        grads = T.grad(tractable_obj, params, consider_constant = constants)

        #put gradients into convenient dictionary
        params_to_grads = {}
        for param, grad in zip(params, grads):
            params_to_grads[param] = grad


        print min_informative_str(params_to_grads[self.dbm.W[0]])
        assert False

        #add the approximate gradients
        params_to_approx_grads = self.dbm.get_neg_phase_grads()

        for param in params_to_approx_grads:
            params_to_grads[param] = params_to_grads[param] + params_to_approx_grads[param]

        learning_updates = self.get_param_updates(params_to_grads)

        sampling_updates = self.dbm.get_sampling_updates()

        for key in sampling_updates:
            learning_updates[key] = sampling_updates[key]

        self.censor_updates(learning_updates)

        print "compiling function..."
        t1 = time.time()
        rval = function([V], updates = learning_updates)
        t2 = time.time()
        print "... compilation took "+str(t2-t1)+" seconds"
        print "graph size: ",len(rval.maker.env.toposort())

        return rval

obj = DebugPDDBM(learning_rate = .01,
        dbm_weight_decay = [ 100. ],
        dbm =  DebugDBM (
                negative_chains = 100,
                monitor_params = 1,
                rbms = [ RBM(
                                                  nvis = 400,
                                                  nhid = 400,
                                                  irange = .05,
                                                  init_bias_vis = -3.
                                                )
                         ]
        ),
        s3c = S3C (
               nvis = 108,
               nhid = 400,
               init_bias_hid = -3.,
               max_bias_hid = -2.,
               min_bias_hid = -8.,
               irange  = .02,
               constrain_W_norm = 1,
               init_B  = 3.,
               min_B   = .1,
               max_B   = 1e6,
               tied_B =  1,
               init_alpha = 1.,
               min_alpha = 1e-3,
               max_alpha = 1e6,
               init_mu =  0.,
               monitor_params = [ 'B', 'p', 'alpha', 'mu', 'W' ],
               m_step =  Grad_M_Step()
               ),
       inference_procedure = DebugInferenceProcedure(
                schedule = [ ['h',1.],   ['h', 0.1],
                             ['h',0.1], ['g',0],   ['h',0.1],  ['h',0.1],
                             ['g',0],   ['h',0.1], ['h', 0.1], ['g',0],
                             ['h',0.1], ['g',0],   ['h',0.1], ['h',0.1] ],
                monitor_kl = 0,
                clip_reflections = 0,
       ),
       print_interval =  10000
)
