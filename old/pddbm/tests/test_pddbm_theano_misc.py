from pylearn2.models.s3c import S3C
from pylearn2.models.s3c import E_Step
from pylearn2.models.s3c import Grad_M_Step
from galatea.pddbm.pddbm_theano import PDDBM, InferenceProcedure
from pylearn2.models.rbm import RBM
from pylearn2.models.dbm import DBM
from pylearn2.utils import as_floatX
from theano import function
import numpy as np
import theano.tensor as T
from theano import config
#from pylearn2.utils import serial
import warnings


if config.floatX != 'float64':
    config.floatX = 'float64'
    warnings.warn("Changed config.floatX to float64. s3c inference tests currently fail due to numerical issues for float32")

def broadcast(mat, shape_0):
    rval = mat
    if mat.shape[0] != shape_0:
        assert mat.shape[0] == 1

        rval = np.zeros((shape_0, mat.shape[1]),dtype=mat.dtype)

        for i in xrange(shape_0):
            rval[i,:] = mat[0,:]

    return rval


class Test_PDDBM_Misc:
    def __init__(self):
        """ gets a small batch of data
            sets up a PD-DBM model
        """

        self.tol = 1e-5

        X = np.random.RandomState([1,2,3]).randn(1000,5)

        X -= X.mean()
        X /= X.std()
        m, D = X.shape
        N = 6
        N2 = 7


        s3c = S3C(nvis = D,
                 nhid = N,
                 irange = .1,
                 init_bias_hid = -1.5,
                 init_B = 3.,
                 min_B = 1e-8,
                 max_B = 1000.,
                 init_alpha = 1., min_alpha = 1e-8, max_alpha = 1000.,
                 init_mu = 1., e_step = None,
                 m_step = Grad_M_Step(),
                 min_bias_hid = -1e30, max_bias_hid = 1e30,
                )

        rbm = RBM(nvis = N, nhid = N2, irange = .1, init_bias_vis = -1.5, init_bias_hid = 1.5)

        #don't give the model an inference procedure or learning rate so it won't spend years compiling a learn_func
        self.model = PDDBM(
                dbm = DBM(  use_cd = 1,
                            rbms = [ rbm  ]),
                s3c = s3c
        )

        self.model.make_pseudoparams()

        self.inference_procedure = InferenceProcedure(
                    schedule = [ ['s',.1],   ['h',.1],   ['g',0, 0.2],   ['h', 0.2], ['s',0.2],
                                ['h',0.3], ['g',0,.3],   ['h',0.3], ['s',0.4], ['h',0.4],
                                ['g',0,.4],   ['h',0.4], ['s',.4], ['h',0.4],
                                ['g',0,.5],   ['h',0.5], ['s', 0.5], ['h',0.1],
                                ['s',0.5] ],
                    clip_reflections = True,
                    rho = .5 )
        self.inference_procedure.register_model(self.model)

        self.X = X
        self.N = N
        self.N2 = N2
        self.m = m


    def test_d_kl_d_h(self):

        "tests that the gradient of the kl with respect to h matches my analytical version of it "

        model = self.model
        ip = self.inference_procedure
        X = self.X

        assert X.shape[0] == self.m

        H = np.cast[config.floatX](self.model.rng.uniform(0.001,.999,(self.m, self.N)))
        S = np.cast[config.floatX](self.model.rng.uniform(-5.,5.,(self.m, self.N)))
        G = np.cast[config.floatX](self.model.rng.uniform(0.001,.999,(self.m,self.N2)))

        H_var = T.matrix(name='H_var')
        H_var.tag.test_value = H
        S_var = T.matrix(name='S_var')
        S_var.tag.test_value = S
        G_var = T.matrix(name='G_var')
        G_var.tag.test_value = G


        sigma0 = ip.infer_var_s0_hat()
        Sigma1 = ip.infer_var_s1_hat()
        mu0 = T.zeros_like(model.s3c.mu)

        #multiply by m to use sum rather than mean
        #otherwise larger batch sizes shrink the gradient so errors appear
        #smaller and larger batch sizes make the unit test easier rather than harder
        #to falsely pass
        trunc_kl = ip.truncated_KL( V = X, obs = { 'H_hat' : H_var,

                                                 'S_hat' : S_var,
                                                 'var_s0_hat' : sigma0,
                                                 'var_s1_hat' : Sigma1,
                                                 'G_hat' : ( G_var, ) } )

        assert len(trunc_kl.type.broadcastable) == 1

        trunc_kl = trunc_kl.sum()

        grad_H = T.grad(trunc_kl, H_var)

        grad_func = function([H_var, S_var, G_var], grad_H)

        grad_theano = grad_func(H,S,G)

        half = as_floatX(.5)
        one = as_floatX(1.)
        two = as_floatX(2.)
        pi = as_floatX(np.pi)
        e = as_floatX(np.e)
        mu = self.model.s3c.mu
        alpha = self.model.s3c.alpha
        W = self.model.s3c.W
        B = self.model.s3c.B
        w = self.model.s3c.w

        term1 = T.log(H_var)
        term2 = -T.log(one - H_var)
        term3 = - half * T.log( Sigma1 *  two * pi * e )
        term4 = half * T.log(sigma0 *  two * pi * e )
        term5 = - self.model.s3c.bias_hid
        assert self.model.s3c.bias_hid is self.model.dbm.bias_vis
        term6 = half * ( - sigma0 + Sigma1 + T.sqr(S_var) )
        term7 = - mu * alpha * S_var
        term8 = half * T.sqr(mu) * alpha
        term9 = - T.dot(X * B, W) * S_var
        term10 = S_var * T.dot(T.dot(H_var * S_var, W.T * B),W)
        term11 = - w * T.sqr(S_var) * H_var
        term12 = half * (Sigma1 + T.sqr(S_var)) * T.dot(B,T.sqr(W))
        term13 = - T.dot( G_var, self.model.dbm.W[0].T )

        analytical = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + term11 + term12 + term13

        grad_analytical = function([H_var, S_var, G_var], analytical)(H,S,G)

        if not np.allclose(grad_theano, grad_analytical):
            print 'grad theano: ',(grad_theano.min(), grad_theano.mean(), grad_theano.max())
            print 'grad analytical: ',(grad_analytical.min(), grad_analytical.mean(), grad_analytical.max())
            ad = np.abs(grad_theano-grad_analytical)
            print 'abs diff: ',(ad.min(),ad.mean(),ad.max())
            assert False

if __name__ == '__main__':
    obj = Test_PDDBM_Misc()

    obj.test_d_kl_d_h()
