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
from pylearn2.models.s3c import damp


class DebugInferenceProcedure(InferenceProcedure):
    def infer(self, V, return_history = False):
        """

            return_history: if True:
                                returns a list of dictionaries with
                                showing the history of the variational
                                parameters
                                throughout fixed point updates
                            if False:
                                returns a dictionary containing the final
                                variational parameters
        """

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

        for Hv in get_debug_values(H_hat):
            if isinstance(Hv, tuple):
                warnings.warn("I got a tuple from this and I have no idea why the fuck that happens. Pulling out the single element of the tuple")
                Hv ,= Hv

            if Hv.shape[1] != s3c_e_step.model.nhid:
                debug_error_message('H prior has wrong # hu, expected %d actual %d'%\
                        (s3c_e_step.model.nhid,
                            Hv.shape[1]))

        for Sv in get_debug_values(S_hat):
            if isinstance(Sv, tuple):
                warnings.warn("I got a tuple from this and I have no idea why the fuck that happens. Pulling out the single element of the tuple")
                Sv ,= Sv

            if Sv.shape[1] != s3c_e_step.model.nhid:
                debug_error_message('prior has wrong # hu, expected %d actual %d'%\
                        (s3c_e_step.model.nhid,
                            Sv.shape[1]))

        def check_H(my_H, my_V):
            if my_H.dtype != config.floatX:
                raise AssertionError('my_H.dtype should be config.floatX, but they are '
                        ' %s and %s, respectively' % (my_H.dtype, config.floatX))

            allowed_v_types = ['float32']

            if config.floatX == 'float64':
                allowed_v_types.append('float64')

            assert my_V.dtype in allowed_v_types

            if config.compute_test_value != 'off':
                from theano.gof.op import PureOp
                Hv = PureOp._get_test_value(my_H)

                Vv = my_V.tag.test_value

                assert Hv.shape[0] == Vv.shape[0]

        check_H(H_hat,V)

        def make_dict():

            return {
                    'G_hat' : tuple(G_hat),
                    'H_hat' : H_hat,
                    'S_hat' : S_hat,
                    'var_s0_hat' : var_s0_hat,
                    'var_s1_hat': var_s1_hat,
                    }

        history = [ make_dict() ]

        for i, step in enumerate(self.schedule):

            letter, number = step

            coeff = as_floatX(number)
            coeff = T.as_tensor_variable(coeff)
            coeff.name = 'coeff_step_'+str(i)

            if letter == 's':
                S_hat = s3c_e_step.infer_S_hat(V, H_hat, S_hat)
                S_hat.name = 'S_hat_step_'+str(i)

            elif letter == 'h':

                new_H = self.infer_H_hat(V = V, H_hat = H_hat, S_hat = S_hat, G1_hat = G_hat[0])

                new_H.name = 'new_H_step_'+str(i)

                H_hat = new_H #damp(old = H_hat, new = new_H, new_coeff = coeff)
                H_hat.name = 'new_H_hat_step_'+str(i)

                check_H(H_hat,V)

            elif letter == 'g':

                if not isinstance(number, int):
                    raise ValueError("Step parameter for 'g' code must be an integer in [0, # g layers) "
                            "but got "+str(number)+" (of type "+str(type(number)))

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

            history.append(make_dict())

        if return_history:
            return history
        else:
            return history[-1]


obj = PDDBM(learning_rate = .01,
        dbm_weight_decay = [ 100. ],
        dbm =  DBM (
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
                schedule = [ ['s',1.],   ['h',1.],   ['g',0],   ['h', 0.1], ['s',0.1],
                             ['h',0.1], ['g',0],   ['h',0.1], ['s',0.1],  ['h',0.1],
                             ['g',0],   ['h',0.1], ['s',0.1], ['h', 0.1], ['g',0],
                             ['h',0.1], ['g',0],   ['h',0.1], ['s', 0.1], ['h',0.1] ],
                monitor_kl = 0,
                clip_reflections = 0,
       ),
       print_interval =  10000
)
