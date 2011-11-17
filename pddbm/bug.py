from galatea.pddbm.pddbm import InferenceProcedure
from pylearn2.models.s3c import S3C
from pylearn2.models.s3c import Grad_M_Step
from theano import config
from pylearn2.utils import make_name, as_floatX
from theano import tensor as T
from theano.printing import min_informative_str
import numpy as np
from pylearn2.utils import sharedX

W = sharedX( np.zeros((400,400)))
W.name = 'W'

class DebugInferenceProcedure(InferenceProcedure):

    def infer(self, V, return_history = False):


        return make_dict()


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

inference_procedure.model = dummy
inference_procedure.s3c_e_step = s3c.e_step

V = T.matrix()
V.name = 'V'

m = V.shape[0]
m.name = 'm'

H_hat = T.matrix('init_H')
G_hat = [ T.matrix('init_G') ]

H_hat.name = 'init_H_hat'

def make_dict():

    return {
            'G_hat' : tuple(G_hat),
            'H_hat' : H_hat,
            }


H_hat = T.dot( G_hat[0] , W)
H_hat.name = 'new_H_hat_step_0'

H_hat_below = H_hat
G_hat[0] = T.nnet.sigmoid(T.dot(H_hat_below, W))
G_hat[0].name = 'new_G_hat_step_1'

constants = set([H_hat, G_hat[0]])

for i, G in enumerate(G_hat):
    G.name = 'final_G_hat[%d]' % (i,)
H_hat.name = 'final_H_hat'

assert H_hat in constants
assert G_hat[0] in constants


a = T.dot(H_hat.T, G_hat[0])
b = T.sum(W * a)


test = T.grad(b, W, consider_constant = constants)
print min_informative_str(test)
assert False
