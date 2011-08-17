from galatea.s3c.s3c import S3C, SufficientStatistics, VHSU_Solve_M_Step
from pylearn2.datasets.cifar10 import CIFAR10
from pylearn2.utils import as_floatX
from theano import function
import numpy as np
import theano.tensor as T
import copy
from theano.printing import Print

tol = 1e-7

class TestS3C:
    def __init__(self):
        self.bSetup = False


    def setup(self):
        """ gets a small batch of data
            sets up an S3C model and learns on the data
            creates an expression for the log likelihood of the data
        """
        dataset = CIFAR10(which_set = 'train')

        X = dataset.get_batch_design(2)
        X = X[:,0:5]
        X -= X.mean()
        X /= X.std()
        m, D = X.shape

        self.model = S3C(nvis = D,
                         nhid = 3,
                         irange = .5,
                         init_bias_hid = 0.,
                         init_B = 3.,
                         min_B = 3.,
                         max_B = 3.,
                         init_alpha = 1., min_alpha = 1e-8, max_alpha = 1000.,
                         init_mu = 1., N_schedule = [1., 2., 4, 8., 16., 32., 64., 128., 200. ],
                         new_stat_coeff = 1.,
                         m_step = VHSU_Solve_M_Step( new_coeff = 1.0 ),
                         W_eps = 1e-6, mu_eps = 1e-8,
                         min_bias_hid = -1e30, max_bias_hid = 1e30,
                        learn_after = None)

        self.orig_params = self.model.get_param_values()

        model = self.model
        mf_obs = model.mean_field(X)

        keys = copy.copy(mf_obs.keys())

        f = function([], [ mf_obs[key] for key in mf_obs ] )

        values = f()

        for key, value in zip(keys,values):
            mf_obs[key] = value

        self.stats = SufficientStatistics.from_observations(X, N = model.nhid,
                B = model.B.get_value(), W = model.W.get_value(), ** mf_obs)


        self.model.learn_mini_batch(X)



        self.prob = self.model.log_likelihood_vhsu( self.stats )

        self.bSetup = True

    def test_grad_vshu_solve_M_step(self):
        """ tests that the learned model has 0 gradient with respect to
            the parameters """
        if not self.bSetup:
            self.setup()

        params = self.model.get_params()

        grads = T.grad(self.prob, params)

        f = function([],grads)

        g = f()

        failing_grads = {}

        for g, param in zip(g,params):
            max_g = np.abs(g).max()

            if max_g > tol:
                failing_grads[param.name] = max_g

        if len(failing_grads.keys()) > 0:
            raise Exception('gradients of log likelihood with respect to parameters should all be 0,'+\
                            ' but the following parameters have the following max abs gradient elem: '+\
                            str(failing_grads) )

    def test_test_setup(self):
        """ tests that the statistics really are frozen, that model parameters don't affect them """

        if not self.bSetup:
            self.setup()

        params = self.model.get_params()

        grads1 = T.grad(self.prob, params)
        f1 = function([], grads1)
        gv1 = f1()

        grads2 = T.grad(self.prob, params, consider_constant = self.stats.d.values() )
        f2 = function([], grads2)
        gv2 = f2()

        fails = {}
        for g1, g2, p in zip(gv1,gv2,params):
            d = np.abs(g1-g2).max()
            if d > tol:
                fails[p.name] = d

        if len(fails.keys()) > 0:
            raise Exception("gradients wrt parameters should not change if " + \
                    " the suff stats are considered constant, but the following "+\
                    " gradients changed by the following amounts: "+str(fails)+\
                    " (this indicates a problem in the testing setup itself) ")





    def test_grad_b(self):
        """ tests that the gradient of the log probability with respect to bias_hid
            matches my analytical derivation """

        if not self.bSetup:
            self.setup()

        print "SETUP DONE"

        g = T.grad(self.prob, self.model.bias_hid)

        mean_h = self.stats.d['mean_h']

        biases = self.model.bias_hid

        sigmoid = T.nnet.sigmoid(biases)


        analytical = mean_h - sigmoid


        f = function([],(g,analytical))

        gv, av = f()

        assert gv.shape == av.shape

        max_diff = np.abs(gv-av).max()

        if max_diff > tol:
            raise Exception("analytical gradient on b deviates from theano gradient on b by up to "+str(max_diff))

        max_av = np.abs(av).max()

        if max_av > tol:
            raise Exception("analytical gradient on b should be 0. max deviation "+str(max_av)+\
                            "average deviation "+str(np.abs(av).mean()))

    def test_grad_B(self):
        """ tests that the gradient of the log probability with respect to B
        matches my analytical derivation """

        if not self.bSetup:
            self.setup()

        g = T.grad(self.prob, self.model.B)

        u_stat_1 = self.stats.d['u_stat_1']
        half = as_floatX(0.5)

        term1 = half * (self.model.W * u_stat_1.T).sum(axis=1)

        mean_hsv = self.stats.d['mean_hsv']

        term2 = (self.model.W * mean_hsv.T).sum(axis=1)

        N = as_floatX(self.model.nhid)

        mean_sq_hs = self.stats.d['mean_sq_hs']

        term3 = - half * N * T.dot(T.sqr(self.model.W), mean_sq_hs)

        fourth = as_floatX(.25)

        u_stat_2 = self.stats.d['u_stat_2']

        term4 = - fourth * u_stat_2

        one = as_floatX(1.)

        term5 = T.sqr(N + one) * half / self.model.B

        analytical = term1 + term2 + term3 + term4 + term5

        f = function([],(g,analytical))

        gv, av = f()

        assert gv.shape == av.shape

        max_diff = np.abs(gv-av).max()

        if max_diff > tol:
            raise Exception("analytical gradient on B deviates from theano gradient on B by up to "+str(max_diff))

    def test_likelihood_vshu_solve_M_step(self):
        """ tests that the log likelihood was increased by the learning """

        if not self.bSetup:
            self.setup()

        f = function([], self.prob)

        new_likelihood = f()

        new_params = self.model.get_param_values()

        self.model.set_param_values(self.orig_params)

        old_likelihood = f()

        self.model.set_param_values(new_params)

        if new_likelihood < old_likelihood:
            raise Exception('M step worsened likelihood. new likelihood: ',new_likelihood,
                ' old likelihood: ', old_likelihood)


if __name__ == '__main__':
    obj = TestS3C()
    obj.test_grad_b()
