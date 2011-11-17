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

class DebugInferenceProcedure(InferenceProcedure):
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
                    'var_s1_hat': var_s1_hat,
                    }

        for i, step in enumerate(self.schedule):

            letter, number = step

            if letter == 'h':

                H_hat = self.infer_H_hat(V = V, H_hat = H_hat, S_hat = S_hat, G1_hat = G_hat[0])
                H_hat.name = 'new_H_hat_step_'+str(i)

            elif letter == 'g':

                b = self.model.dbm.bias_hid[number]

                W = self.model.dbm.W

                W_below = W[number]

                if number == 0:
                    H_hat_below = H_hat
                else:
                    H_hat_below = G_hat[number - 1]

                num_g = self.model.num_g

                if number == num_g - 1:
                    G_hat[number] = dbm_ip.infer_H_hat_one_sided(other_H_hat = H_hat_below, W = W_below, b = b)
                else:
                    H_hat_above = G_hat[number + 1]
                    W_above = W[number+1]
                    G_hat[number] = dbm_ip.infer_H_hat_two_sided(H_hat_below = H_hat_below, H_hat_above = H_hat_above,
                                           W_below = W_below, W_above = W_above,
                                           b = b)
            else:
                raise ValueError("Invalid inference step code '"+letter+"'. Valid options are 's','h' and 'g'.")


        return make_dict()


class DebugDBM(DBM):
    def expected_energy(self, V_hat, H_hat):
        V_name = make_name(V_hat, 'anon_V_hat')
        assert isinstance(H_hat, (list,tuple))

        H_names = []
        for i in xrange(len(H_hat)):
            H_names.append( make_name(H_hat[i], 'anon_H_hat[%d]' %(i,) ))

        m = V_hat.shape[0]
        m.name = V_name + '.shape[0]'

        assert len(H_hat) == len(self.rbms)

        v = T.mean(V_hat, axis=0)

        v_bias_contrib = T.dot(v, self.bias_vis)

        exp_vh = T.dot(V_hat.T,H_hat[0]) / m

        v_weights_contrib = T.sum(self.W[0] * exp_vh)

        v_weights_contrib.name = 'v_weights_contrib('+V_name+','+H_names[0]+')'

        total = v_bias_contrib + v_weights_contrib

        highest_bias_contrib = T.dot(T.mean(H_hat[-1],axis=0), self.bias_hid[-1])

        total = total + highest_bias_contrib

        assert len(total.type.broadcastable) == 0

        rval =  - total

        return rval



obj = PDDBM(learning_rate = .01,
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
                schedule = [ ['h',1.],   ['g',0],   ['h', 0.1],
                             ['h',0.1], ['g',0],   ['h',0.1],  ['h',0.1],
                             ['g',0],   ['h',0.1], ['h', 0.1], ['g',0],
                             ['h',0.1], ['g',0],   ['h',0.1], ['h',0.1] ],
                monitor_kl = 0,
                clip_reflections = 0,
       ),
       print_interval =  10000
)
