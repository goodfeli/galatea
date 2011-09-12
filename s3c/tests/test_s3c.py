import warnings
from galatea.s3c.s3c import S3C
from galatea.s3c.s3c import SufficientStatistics
from galatea.s3c.s3c import SufficientStatisticsHolder
from galatea.s3c.s3c import VHS_E_Step
from galatea.s3c.s3c import VHS_Solve_M_Step
from pylearn2.utils import as_floatX
from theano import function
import numpy as np
import theano.tensor as T
import copy
from theano.printing import Print
from theano import config
from pylearn2.utils import serial
from theano.sandbox.linalg.ops import alloc_diag, extract_diag, matrix_inverse
from matplotlib import pyplot as plt

class TestS3C_VHS:

    def trace_out_B(self, X, holder):
        """ this function was used for debugging the M step B update but is no longer used in the tests
        this function should be called AFTER calls to learn_minibatch or it won't be using the same parameters as the other tests
        """

        model = self.model
        mf_obs = model.e_step.mean_field(X)

        print '------wtf stats----------------------'
        for key in holder.d:
            print key,': ',holder.d[key].get_value()
        print '------------------------------------'

        stats = SufficientStatistics.from_holder(holder)

        orig_B = model.B_driver.get_value()

        delta = .001
        B = np.arange(delta,5.,delta)

        obj = np.zeros(B.shape)

        f = function([], model.expected_log_prob_vhs(stats))

        for i in xrange(B.shape[0]):
            #print i
            model.B_driver.set_value(np.cast[config.floatX](B[i]))
            obj[i] = f()

        print 'optimal B',B[obj==obj.max()]

        plt.plot(B,obj)
        plt.show()

        model.B_driver.set_value(orig_B)

    def __init__(self):
        """ gets a small batch of data
            sets up an S3C model and learns on the data
            creates an expression for the log likelihood of the data
        """

        self.tol = 1e-5

        dataset = serial.load('${GOODFELI_TMP}/cifar10_preprocessed_train_1K.pkl')

        X = dataset.get_batch_design(1000)
        #X = X[:,0:2]
        #warnings.warn('hack')
        #X[0,0] = 1.
        #X[0,1] = -1.
        m, D = X.shape
        N = 300

        self.model = S3C(nvis = D,
                #disable_W_update = 1,
                         nhid = N,
                         irange = .5,
                         init_bias_hid = 0.,
                         init_B = 1.,
                         min_B = 1e-8,
                         max_B = 1e8,
                         tied_B = 1,
                         e_step = VHS_E_Step(
                             #h_new_coeff_schedule = [ ],
                             h_new_coeff_schedule = [ .01 ]
                         ),
                         init_alpha = 1.,
                         min_alpha = 1e-8, max_alpha = 1e8,
                         init_mu = 1.,
                         new_stat_coeff = 1.,
                         m_step = VHS_Solve_M_Step( new_coeff = 1.0 ),
                         W_eps = 0., mu_eps = 0.,
                        learn_after = None)

        #warnings.warn('hack')
        #W = self.model.W.get_value()
        #W[0,0] = 1.
        #W[1,0] = 1.
        #self.model.W.set_value(W)

        self.orig_params = self.model.get_param_values()

        model = self.model
        mf_obs = model.e_step.mean_field(X)

        stats = SufficientStatistics.from_observations(needed_stats =
                model.m_step.needed_stats(), X =X,
                N = model.nhid, B = model.get_B_value(),
                W = model.W.get_value(), ** mf_obs)

        holder = SufficientStatisticsHolder(
                needed_stats = model.m_step.needed_stats(),
                nvis = D, nhid = N)

        keys = copy.copy(stats.d.keys())

        outputs = [ stats.d[key] for key in keys ]

        f = function([], outputs)

        vals = f()

        for key, val in zip(keys, vals):
            holder.d[key].set_value(val)


        self.stats = SufficientStatistics.from_holder(holder)


        self.model.learn_mini_batch(X)

        #self.trace_out_B(X, holder)

        self.new_params = model.get_param_values()


        self.prob = self.model.expected_log_prob_vhs( self.stats )


    def test_grad_vsh_solve_M_step(self):
        """ tests that the learned model has 0 gradient with respect to
            the parameters """

        params = self.model.get_params()

        grads = T.grad(self.prob, params)

        f = function([],grads)

        g = f()

        failing_grads = {}

        for g, param in zip(g,params):
            max_g = np.abs(g).max()

            if max_g > self.tol:
                if len(g.shape) == 0:
                    failing_grads[param.name] = g
                else:
                    failing_grads[param.name] = g[np.abs(g) == max_g]

        if len(failing_grads.keys()) > 0:
            raise Exception('gradients of log likelihood with respect to parameters should all be 0,'+\
                            ' but the following parameters have the following max abs gradient elem: '+\
                            str(failing_grads) )



    def test_W_jump(self):

        " tests that W is where I think it should be "

        stats = self.stats

        cov_hs = stats.d['second_hs']
        assert cov_hs.dtype == config.floatX
        #mean_hsv[i,j] = E_D,Q h_i s_i v_j
        mean_hsv = stats.d['mean_hsv']

        regularized = cov_hs + alloc_diag(T.ones_like(self.model.mu) * self.model.W_eps)
        assert regularized.dtype == config.floatX


        inv = matrix_inverse(regularized)
        assert inv.dtype == config.floatX

        new_W = T.dot(mean_hsv.T, inv)

        f = function([], new_W)

        Wv = f()
        aWv = self.model.W.get_value()

        diffs = Wv - aWv
        max_diff = np.abs(diffs).max()

        if max_diff > self.tol:
            raise Exception("W deviates from its correct value by at most "+str(max_diff))


    def test_test_setup(self):
        """ tests that the statistics really are frozen, that model parameters don't affect them """


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
            if d > self.tol:
                fails[p.name] = d

        if len(fails.keys()) > 0:
            raise Exception("gradients wrt parameters should not change if " + \
                    " the suff stats are considered constant, but the following "+\
                    " gradients changed by the following amounts: "+str(fails)+\
                    " (this indicates a problem in the testing setup itself) ")


    """def test_grad_b(self):
        "" tests that the gradient of the log probability with respect to bias_hid
            matches my analytical derivation ""


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

        if max_diff > self.tol:
            raise Exception("analytical gradient on b deviates from theano gradient on b by up to "+str(max_diff))

        max_av = np.abs(av).max()
"""

    def test_grad_alpha(self):
        """tests that the gradient of the log probability with respect to alpha
        matches my analytical derivation """

        self.model.set_param_values(self.new_params)

        g = T.grad(self.prob, self.model.alpha)

        mu = self.model.mu
        alpha = self.model.alpha
        half = as_floatX(.5)

        mean_sq_s = self.stats.d['mean_sq_s']
        mean_hs = self.stats.d['mean_hs']
        mean_h = self.stats.d['mean_h']

        term1 = - half * mean_sq_s

        term2 = mu * mean_hs

        term3 = - half * T.sqr(mu) * mean_h

        term4 = half / alpha

        analytical = term1 + term2 + term3 + term4

        f = function([],(g,analytical))

        gv, av = f()

        assert gv.shape == av.shape

        max_diff = np.abs(gv-av).max()

        if max_diff > self.tol:
            print "gv"
            print gv
            print "av"
            print av
            raise Exception("analytical gradient on alpha deviates from theano gradient on alpha by up to "+str(max_diff))

    def test_grad_W(self):
        """tests that the gradient of the log probability with respect to W
        matches my analytical derivation """

        self.model.set_param_values(self.new_params)

        g = T.grad(self.prob, self.model.W)

        B = self.model.B
        W = self.model.W
        mean_hsv = self.stats.d['mean_hsv']
        second_hs = self.stats.d['second_hs']

        term1 = (B * mean_hsv).T
        term2 = - B.dimshuffle(0,'x') * T.dot(W, second_hs)

        analytical = term1 + term2

        f = function([],(g,analytical))

        gv, av = f()

        assert gv.shape == av.shape

        max_diff = np.abs(gv-av).max()

        if max_diff > self.tol:
            print "gv"
            print gv
            print "av"
            print av
            raise Exception("analytical gradient on W deviates from theano gradient on W by up to "+str(max_diff))


    def test_alpha_jump(self):


        " tests that alpha is where I think it should be "

        stats = self.stats

        mean_h = stats.d['mean_h']
        new_mu = self.model.mu
        mean_hs = stats.d['mean_hs']
        mean_sq_s = stats.d['mean_sq_s']

        one = as_floatX(1.)
        two = as_floatX(2.)
        s_denom1 = mean_sq_s
        s_denom2 = - two * new_mu * mean_hs
        s_denom3 = T.sqr(new_mu) * mean_h

        s_denom = s_denom1 + s_denom2 + s_denom3
        new_alpha =  one / s_denom
        new_alpha.name = 'new_alpha'

        f = function([], new_alpha)

        Alphav = f()
        aAlphav = self.model.alpha.get_value()


        diffs = Alphav - aAlphav
        max_diff = np.abs(diffs).max()

        if max_diff > self.tol:
            print 'Actual alpha: '
            print aAlphav
            print 'Expected alpha: '
            print Alphav
            raise Exception("alpha deviates from its correct value by at most "+str(max_diff))



    def test_likelihood_vsh_solve_M_step(self):
        """ tests that the log likelihood was increased by the learning """

        f = function([], self.prob)

        new_likelihood = f()

        if np.isnan(new_likelihood) or np.isinf(new_likelihood):
            raise Exception('new_likelihood is NaN/Inf')


        self.model.set_param_values(self.orig_params)

        old_likelihood = f()

        self.model.set_param_values(self.new_params)

        if np.isnan(old_likelihood) or np.isinf(old_likelihood):
            raise Exception('old_likelihood is NaN/Inf')


        if new_likelihood < old_likelihood:
            raise Exception('M step worsened likelihood. new likelihood: ',new_likelihood,
                ' old likelihood: ', old_likelihood)


if __name__ == '__main__':
    obj = TestS3C_VHS()
    obj.test_grad_vsh_solve_M_step()
    #obj.test_likelihood_vsh_solve_M_step()
