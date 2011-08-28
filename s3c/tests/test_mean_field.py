from galatea.s3c.s3c import S3C
from galatea.s3c.s3c import SufficientStatistics
from galatea.s3c.s3c import SufficientStatisticsHolder
from galatea.s3c.s3c import VHSU_E_Step
from galatea.s3c.s3c import VHS_E_Step
from galatea.s3c.s3c import VHSU_Solve_M_Step
from galatea.s3c.s3c import VHS_Solve_M_Step
from pylearn2.datasets.cifar10 import CIFAR10
from pylearn2.utils import as_floatX
from theano import function
import numpy as np
import theano.tensor as T
import copy
from theano.printing import Print
from theano import config
from pylearn2.utils import serial
from theano.sandbox.linalg.ops import alloc_diag, extract_diag, matrix_inverse
config.compute_test_value = 'raise'

class TestMeanField_VHS:
    def __init__(self):
        """ gets a small batch of data
            sets up an S3C model and learns on the data
            creates an expression for the log likelihood of the data
        """

        self.tol = 1e-5

        dataset = serial.load('/data/lisatmp/goodfeli/cifar10_preprocessed_train_2M.pkl')

        X = dataset.get_batch_design(1000)
        X = X[:,0:5]
        X -= X.mean()
        X /= X.std()
        m, D = X.shape
        N = 5

        self.model = S3C(nvis = D,
                         nhid = N,
                         irange = .5,
                         init_bias_hid = 0.,
                         init_B = 3.,
                         min_B = 1e-8,
                         max_B = 1000.,
                         init_alpha = 1., min_alpha = 1e-8, max_alpha = 1000.,
                         init_mu = 1., e_step = VHS_E_Step(h_new_coeff_schedule = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1. ]),
                         new_stat_coeff = 1.,
                         m_step = VHS_Solve_M_Step( new_coeff = 1.0 ),
                         W_eps = 0., mu_eps = 1e-8,
                         min_bias_hid = -1e30, max_bias_hid = 1e30,
                        learn_after = None)

        self.X = X
        self.N = N


    def test_grad_s(self):

        "tests that the gradients with respect to s_i are 0 after doing a mean field update of s_i "

        model = self.model
        e_step = model.e_step
        X = self.X

        init_H = e_step.init_mf_H(V = X)
        init_Mu1 = e_step.init_mf_Mu1(V = X)

        H, Mu1 = function([], outputs=[init_H, init_Mu1])()

        H_var = T.matrix()
        Mu1_var = T.matrix()
        idx = T.iscalar()

        A = e_step.mean_field_A(V = X, H = H_var, Mu1 = Mu1_var)

        S = e_step.mean_field_Mu1(A = A)

        s_idx = S[:,idx]

        s_i_func = function([H_var,Mu1_var,idx],s_idx)

        sigma0 = 1. / model.alpha
        Sigma1 = e_step.mean_field_Sigma1()
        mu0 = T.zeros_like(model.mu)

        #by truncated KL, I mean that I am dropping terms that don't depend on H and Mu1
        # (they don't affect the outcome of this test and some of them are intractable )
        trunc_kl = - model.entropy_hs(H = H_var, sigma0 = sigma0, Sigma1 = Sigma1) + \
                     model.expected_energy_vhs(V = X, H = H_var, Mu1 = Mu1_var, mu0 = mu0, sigma0 = sigma0, Sigma1 = Sigma1)

        grad_Mu1 = T.grad(trunc_kl.sum(), Mu1_var)

        grad_Mu1_idx = grad_Mu1[:,idx]

        grad_func = function([H_var, Mu1_var, idx], grad_Mu1_idx)

        for i in xrange(self.N):
            Mu1[:,i] = s_i_func(H, Mu1, i)

            g = grad_func(H,Mu1,i)

            assert not np.any(np.isnan(g))

            g_abs_max = np.abs(g).max()

            if g > self.tol:
                raise Exception('after mean field step, gradient of kl divergence wrt mean field parameter should be 0, but here the max magnitude of a gradient element is '+str(g_abs_max)+' after updating s_'+str(i))

if __name__ == '__main__':
    obj = TestMeanField_VHS()

    obj.test_grad_s()
