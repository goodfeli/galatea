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
    def infer_H_hat(self, V, H_hat, G1_hat):
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
        #S_hat = s3c_e_step.init_S_hat(V)

        H_hat.name = 'init_H_hat'
        #S_hat.name = 'init_S_hat'

        def make_dict():

            return {
                    'G_hat' : tuple(G_hat),
                    'H_hat' : H_hat,
                    }

        for i, step in enumerate(self.schedule):
            letter, number = step

            if letter == 'h':
                H_hat = self.infer_H_hat(V = V, H_hat = H_hat, G1_hat = G_hat[0])
                H_hat.name = 'new_H_hat_step_'+str(i)
            elif letter == 'g':
                b = self.model.dbm.bias_hid[0]
                W_below = self.model.dbm.W[0]
                H_hat_below = H_hat
                G_hat[number] = dbm_ip.infer_H_hat_one_sided(other_H_hat = H_hat_below, W = W_below, b = b)

        return make_dict()

class DebugDBM(DBM):
    pass


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
        )

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
               )

s3c.make_pseudoparams()

inference_procedure = DebugInferenceProcedure(
                schedule = [ ['h',1.],   ['g', 0],
                              ],
                monitor_kl = 0,
                clip_reflections = 0,
       )

class A:
    pass

dummy = A()
dummy.s3c = s3c
dummy.dbm = dbm

inference_procedure.model = dummy
inference_procedure.s3c_e_step = s3c.e_step
inference_procedure.dbm_ip = dbm.inference_procedure

V = T.matrix()
V.name = 'V'

m = V.shape[0]
m.name = 'm'

hidden_obs = inference_procedure.infer(V)

constants = set(hidden_obs.values())

G_hat = hidden_obs['G_hat']
for i, G in enumerate(G_hat):
    G.name = 'final_G_hat[%d]' % (i,)
H_hat = hidden_obs['H_hat']
H_hat.name = 'final_H_hat'

assert H_hat in constants
assert G_hat in constants


a = T.dot(H_hat.T, G_hat[0])
b = T.sum(dbm.W[0] * a)

test = T.grad(b, dbm.W[0], consider_constant = constants)
print min_informative_str(test)
assert False
