from pylearn2.models.model import Model
from theano import config, function, shared
import theano.tensor as T
import numpy as np
from theano.sandbox.linalg.ops import alloc_diag, extract_diag, matrix_inverse
from theano.printing import Print
from pylearn2.utils import sharedX, as_floatX
from pylearn2.monitor import Monitor
import copy
#config.compute_test_value = 'raise'

class SufficientStatisticsHolder:
    def __init__(self, nvis, nhid, needed_stats):
        d = {
                    "mean_h"                :   sharedX(np.zeros(nhid), "mean_h" ),
                    "mean_v"                :   sharedX(np.zeros(nvis), "mean_v" ),
                    "mean_sq_v"             :   sharedX(np.zeros(nvis), "mean_sq_v" ),
                    "mean_s1"               :   sharedX(np.zeros(nhid), "mean_s1"),
                    "mean_s"                :   sharedX(np.zeros(nhid), "mean_s" ),
                    "mean_sq_s"             :   sharedX(np.zeros(nhid), "mean_sq_s" ),
                    "mean_hs"               :   sharedX(np.zeros(nhid), "mean_hs" ),
                    "mean_sq_hs"            :   sharedX(np.zeros(nhid), "mean_sq_hs" ),
                    #"mean_D_sq_mean_Q_hs"   :   sharedX(np.zeros(nhid), "mean_D_sq_mean_Q_hs"),
                    "cov_hs"                :   sharedX(np.zeros((nhid,nhid)), 'cov_hs'),
                    "mean_hsv"              :   sharedX(np.zeros((nhid,nvis)), 'mean_hsv'),
                    "u_stat_1"              :   sharedX(np.zeros((nhid,nvis)), 'u_stat_1'),
                    "u_stat_2"              :   sharedX(np.zeros((nvis,)),'u_stat_2')
                }

        self.d = {}

        for stat in needed_stats:
            self.d[stat] = d[stat]


    def __getstate__(self):
        rval = {}

        for name in self.d:
            rval[name] = self.d[name].get_value(borrow=False)

        return rval

    def __setstate__(self,d):
        self.d = {}

        for name in d:
            self.d[name] = shared(d[name])

    def update(self, updates, updated_stats):
        for key in updated_stats.d:
            assert key in self.d
        for key in self.d:
            assert key in updated_stats.d
            assert key not in updates
            updates[self.d[key]] = updated_stats.d[key]

class SufficientStatistics:
    def __init__(self, d):
        self. d = {}
        for key in d:
            self.d[key] = d[key]
        #
    #

    @classmethod
    def from_holder(self, holder):
        return SufficientStatistics(holder.d)


    @classmethod
    def from_observations(self, needed_stats, X, H, mu0, Mu1, sigma0, Sigma1, \
            U = None, N = None, B = None, W = None):

        m = T.cast(X.shape[0],config.floatX)

        #mean_h
        assert H.dtype == config.floatX
        mean_h = T.mean(H, axis=0)
        assert H.dtype == mean_h.dtype
        assert mean_h.dtype == config.floatX

        #mean_v
        mean_v = T.mean(X,axis=0)

        #mean_sq_v
        mean_sq_v = T.mean(T.sqr(X),axis=0)

        #mean_s
        mean_S = H * Mu1 + (1.-H)*mu0
        mean_s = T.mean(mean_S,axis=0)

        #mean_s1
        mean_s1 = T.mean(Mu1,axis=0)

        #mean_sq_s
        mean_sq_S = H * (Sigma1 + T.sqr(Mu1)) + (1. - H)*(sigma0+T.sqr(mu0))
        mean_sq_s = T.mean(mean_sq_S,axis=0)

        #mean_hs
        mean_HS = H * Mu1
        mean_hs = T.mean(mean_HS,axis=0)
        mean_D_sq_mean_Q_hs = T.mean(T.sqr(mean_HS), axis=0)

        #mean_sq_hs
        mean_sq_HS = H * (Sigma1 + T.sqr(Mu1))
        mean_sq_hs = T.mean(mean_sq_HS, axis=0)

        #cov_hs
        outer = T.dot(mean_HS.T,mean_HS) /m
        mask = T.identity_like(outer)
        cov_hs = (1.-mask) * outer + alloc_diag(mean_sq_hs)

        #mean_hsv
        mean_hsv = T.dot(mean_HS.T,X) / m

        u_stat_1 = None
        u_stat_2 = None
        if U is not None:
            N = as_floatX(N)
            #u_stat_1
            two = np.cast[config.floatX](2.)
            u_stat_1 = - two * T.mean( T.as_tensor_variable(mean_HS).dimshuffle(0,1,'x') * U, axis=0)

            #u_stat_2
            #B = Print('B',attrs=['mean'])(B)
            #N = Print('N')(N)
            coeff = two * T.sqr(N)
            #coeff = Print('coeff')(coeff)
            term1 = coeff/B
            #term1 = Print('us2 term1',attrs=['mean'])(term1)
            term2 = two * N * T.dot(T.sqr(mean_HS),T.sqr(W.T))
            #term2 = Print('us2 term2',attrs=['mean'])(term2)
            term3 = - two * T.sqr(T.dot(mean_HS, W.T))
            #term3 = Print('us2 term3',attrs=['mean'])(term3)

            u_stat_2 = (term1+term2+term3).mean(axis=0)



        d = {
                    "mean_h"                :   mean_h,
                    "mean_v"                :   mean_v,
                    "mean_sq_v"             :   mean_sq_v,
                    "mean_s"                :   mean_s,
                    "mean_s1"               :   mean_s1,
                    "mean_sq_s"             :   mean_sq_s,
                    "mean_hs"               :   mean_hs,
                    "mean_sq_hs"            :   mean_sq_hs,
                    #"mean_D_sq_mean_Q_hs"   :   mean_D_sq_mean_Q_hs,
                    "cov_hs"                :   cov_hs,
                    "mean_hsv"              :   mean_hsv,
                    "u_stat_1"              :   u_stat_1,
                    "u_stat_2"              :   u_stat_2
                }


        final_d = {}

        for stat in needed_stats:
            final_d[stat] = d[stat]
            final_d[stat].name = 'observed_'+stat

        return SufficientStatistics(final_d)

    def decay(self, coeff):
        rval_d = {}

        coeff = np.cast[config.floatX](coeff)

        for key in self.d:
            rval_d[key] = self.d[key] * coeff
            rval_d[key].name = 'decayed_'+self.d[key].name
        #

        return SufficientStatistics(rval_d)

    def accum(self, new_stat_coeff, new_stats):

        if hasattr(new_stat_coeff,'dtype'):
            assert new_stat_coeff.dtype == config.floatX
        else:
            assert isinstance(new_stat_coeff,float)
            new_stat_coeff = np.cast[config.floatX](new_stat_coeff)

        rval_d = {}

        for key in self.d:
            rval_d[key] = self.d[key] + new_stat_coeff * new_stats.d[key]
            rval_d[key].name = 'blend_'+self.d[key].name+'_'+new_stats.d[key].name

        return SufficientStatistics(rval_d)


class S3C(Model):
    def __init__(self, nvis, nhid, irange, init_bias_hid,
                       init_B, min_B, max_B,
                       init_alpha, min_alpha, max_alpha, init_mu,
                       new_stat_coeff,
                       e_step,
                       m_step,
                       W_eps = 1e-6, mu_eps = 1e-8,
                        min_bias_hid = -1e30, max_bias_hid = 1e30,
                       learn_after = None, hard_max_step = None):
        """"
        nvis: # of visible units
        nhid: # of hidden units
        irange: (scalar) weights are initinialized ~U( [-irange,irange] )
        init_bias_hid: initial value of hidden biases (scalar or vector)
        init_B: initial value of B (scalar or vector)
        min_B, max_B: (scalar) learning updates to B are clipped to [min_B, max_B]
        init_alpha: initial value of alpha (scalar or vector)
        min_alpha, max_alpha: (scalar) learning updates to alpha are clipped to [min_alpha, max_alpha]
        init_mu: initial value of mu (scalar or vector)
        new_stat_coeff: Exponential decay steps on a variable eta take the form
                        eta:=  new_stat_coeff * new_observation + (1-new_stat_coeff) * eta
        e_step:      An E_Step object that determines what kind of E-step to do
        m_step:      An M_Step object that determines what kind of M-step to do
        W_eps:       L2 regularization parameter for linear regression problem for W
        mu_eps:      L2 regularization parameter for linear regression problem for b
        learn_after: only applicable when new_stat_coeff < 1.0
                        begins learning parameters and decaying sufficient statistics
                        after seeing learn_after examples
                        until this time, only accumulates sufficient statistics
        hard_max_step:  if set to None, has no effect
                        otherwise, every element of every parameter is not allowed to change
                        by more than this amount on each M-step. This is basically a hack
                        introduced to prevent explosion in gradient descent.
        """

        super(S3C,self).__init__()

        self.W_eps = np.cast[config.floatX](float(W_eps))
        self.mu_eps = np.cast[config.floatX](float(mu_eps))
        self.nvis = nvis
        self.nhid = nhid
        self.irange = irange
        self.init_bias_hid = init_bias_hid
        self.init_alpha = init_alpha
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
        self.init_B = init_B
        self.min_B = float(min_B)
        self.max_B = float(max_B)
        self.e_step = e_step
        self.e_step.register_model(self)
        self.m_step = m_step
        self.init_mu = init_mu
        self.min_bias_hid = min_bias_hid
        self.max_bias_hid = max_bias_hid

        self.hard_max_step = hard_max_step
        if self.hard_max_step is not None:
            self.hard_max_step = as_floatX(float(self.hard_max_step))


        #this class always needs a monitor, since it is used to implement the learn_after feature
        Monitor.get_monitor(self)

        self.new_stat_coeff = np.cast[config.floatX](float(new_stat_coeff))
        if self.new_stat_coeff < 1.0:
            assert learn_after is not None
        else:
            assert learn_after is None
        #

        self.learn_after = learn_after

        self.reset_rng()

        self.redo_everything()

    def reset_rng(self):
        self.rng = np.random.RandomState([1.,2.,3.])

    def redo_everything(self):

        self.W = sharedX(self.rng.uniform(-self.irange, self.irange, (self.nvis, self.nhid)), name = 'W')
        self.bias_hid = sharedX(np.zeros(self.nhid)+self.init_bias_hid, name='bias_hid')
        self.alpha = sharedX(np.zeros(self.nhid)+self.init_alpha, name = 'alpha')
        self.mu = sharedX(np.zeros(self.nhid)+self.init_mu, name='mu')
        self.B = sharedX(np.zeros(self.nvis)+self.init_B, name='B')

        if self.new_stat_coeff < 1.0:
            self.suff_stat_holder = SufficientStatisticsHolder(nvis = self.nvis, nhid = self.nhid,
                    needed_stats = self.m_step.needed_stats() )

        self.redo_theano()
    #


    def get_monitoring_channels(self, V):
        return self.m_step.get_monitoring_channels(V, self)

    def get_params(self):
        return [self.W, self.bias_hid, self.alpha, self.mu, self.B ]

    @classmethod
    def solve_vhs_needed_stats(cls):
        return set([ 'cov_hs',
                 'mean_hsv',
                 'mean_v',
                 'mean_sq_v',
                 'mean_sq_s',
                 'mean_sq_hs',
                 'mean_hs',
                 'mean_h'
                ])

    def solve_vhs_from_stats(self, stats):

        """Solve multiple linear regression problem where
         W is a matrix used to predict v from h*s


        cov_hs[i,j] = E_D,Q h_i s_i h_j s_j   (note that diagonal has different formula)
        """

        cov_hs = stats.d['cov_hs']
        assert cov_hs.dtype == config.floatX
        #mean_hsv[i,j] = E_D,Q h_i s_i v_j
        mean_hsv = stats.d['mean_hsv']

        regularized = cov_hs + alloc_diag(T.ones_like(self.mu) * self.W_eps)
        assert regularized.dtype == config.floatX


        inv = matrix_inverse(regularized)
        assert inv.dtype == config.floatX

        new_W = T.dot(inv,mean_hsv).T
        assert new_W.dtype == config.floatX

        #Solve for B by setting gradient of log likelihood to 0
        mean_sq_v = stats.d['mean_sq_v']

        one = as_floatX(1.)
        two = as_floatX(2.)

        denom1 = mean_sq_v
        denom2 = - two * (new_W * mean_hsv.T).sum(axis=1)
        denom3 = (cov_hs.dimshuffle('x',0,1)*new_W.dimshuffle(0,1,'x')*new_W.dimshuffle(0,1,'x')).sum(axis=(1,2))

        denom = T.clip(denom1 + denom2 + denom3, 1e-10, 1e8)

        new_B = one / denom


        mean_hs = stats.d['mean_hs']

        # Now a linear regression problem where mu_i is used to predict
        # s_i from h_i

        # mu_i = ( h^T h + reg)^-1 h^T s_i

        mean_h = stats.d['mean_h']
        assert mean_h.dtype == config.floatX
        reg = self.mu_eps
        mu = mean_hs/(mean_h+reg)


        mean_sq_s = stats.d['mean_sq_s']
        mean_sq_hs = stats.d['mean_sq_hs']

        s_denom1 = mean_sq_s
        s_denom2 = - two * self.mu * mean_hs
        s_denom3 = T.sqr(self.mu) * mean_sq_hs


        s_denom = s_denom1 + s_denom2 + s_denom3

        alpha = one / s_denom


        #probability of hiddens just comes from sample counting
        #to put it back in bias_hid space apply sigmoid inverse

        p = T.clip(mean_h,np.cast[config.floatX](1e-8),np.cast[config.floatX](1.-1e-8))
        p.name = 'mean_h_clipped'

        assert p.dtype == config.floatX

        bias_hid = T.log( - p / (p-1.) )

        assert bias_hid.dtype == config.floatX

        return new_W, bias_hid, alpha, mu, new_B

    def solve_vhsu_from_stats(self, stats):
         #TODO: write unit test verifying that this results in zero gradient

        #Solve for W
        mean_hsv = stats.d['mean_hsv']
        half = np.cast[config.floatX](0.5)
        u_stat_1 = stats.d['u_stat_1']
        mean_sq_hs = stats.d['mean_sq_hs']
        N = np.cast[config.floatX](self.nhid)

        numer1 = mean_hsv.T
        numer2 = half * u_stat_1.T

        numer = numer1 + numer2

        #mean_sq_hs = Print('mean_sq_hs',attrs=['mean'])(mean_sq_hs)

        denom = N * mean_sq_hs

        new_W = numer / denom
        new_W.name = 'new_W'


        #Solve for mu
        mean_hs = stats.d['mean_hs']
        mean_h =  stats.d['mean_h']
        mean_h = Print('mean_h',attrs=['min','mean','max'])(mean_h)
        new_mu = mean_hs / (mean_h + self.W_eps)
        new_mu.name = 'new_mu'


        #Solve for bias_hid
        denom = T.clip(mean_h - 1., -1., -1e-10)


        new_bias_hid = T.log( - mean_h / denom )
        new_bias_hid.name = 'new_bias_hid'


        #Solve for alpha
        mean_sq_s = stats.d['mean_sq_s']
        one = np.cast[config.floatX](1.)
        two = np.cast[config.floatX](2.)
        denom = mean_sq_s + mean_h * T.sqr(new_mu) - two * new_mu * mean_hs
        new_alpha =  one / denom
        new_alpha.name = 'new_alpha'


        #Solve for B
        #new_W = Print('new_W',attrs=['mean'])(new_W)

        numer = T.sqr(N)+one
        numer = Print('numer')(numer)
        assert numer.dtype == config.floatX
        u_stat_2 = stats.d['u_stat_2']
        #u_stat_2 = Print('u_stat_2',attrs=['mean'])(u_stat_2)

        mean_sq_v = stats.d['mean_sq_v']
        #mean_sq_v = Print('mean_sq_v',attrs=['mean'])(mean_sq_v)

        mean_sq_hs = Print('mean_sq_hs',attrs=['mean'])(mean_sq_hs)
        #mean_hsv = Print('mean_hsv',attrs=['mean'])(mean_hsv)

        denom1 = N * T.dot(T.sqr(new_W), mean_sq_hs)
        denom2 = half * u_stat_2
        denom3 = - (new_W.T *  u_stat_1).sum(axis=0)
        denom4 = - two * (new_W.T * mean_hsv).sum(axis=0)
        denom5 = mean_sq_v

        denom = T.clip(denom1 + denom2 + denom3 + denom4 + denom5, 1e-8, 1e12)
        #denom = Print('denom', attrs=['min','max'])(denom)
        assert denom.dtype == config.floatX

        new_B = numer / denom
        new_B.name = 'new_B'
        assert new_B.dtype == config.floatX

        return new_W, new_bias_hid, new_alpha, new_mu, new_B


    def make_learn_func(self, X, learn = None):
        """
        X: a symbolic design matrix
        learn:
            must be None unless using sufficient statistics decay
            False: accumulate sufficient statistics
            True: exponentially decay sufficient statistics, accumulate new ones, and learn new params
        """

        #E step
        hidden_obs = self.e_step.mean_field(X)

        m = T.cast(X.shape[0],dtype = config.floatX)
        N = np.cast[config.floatX](self.nhid)
        new_stats = SufficientStatistics.from_observations(needed_stats = self.m_step.needed_stats(),
                X = X, N = N, B = self.B, W = self.W, **hidden_obs)


        if self.new_stat_coeff == 1.0:
            assert learn is None
            updated_stats = new_stats
            do_learn_updates = True
            do_stats_updates = False
        else:
            do_stats_updates = True
            do_learn_updates = learn

            old_stats = SufficientStatistics.from_holder(self.suff_stat_holder)

            if learn:
                updated_stats = old_stats.decay(1.0-self.new_stat_coeff)
                updated_stats = updated_stats.accum(new_stat_coeff = self.new_stat_coeff, new_stats = new_stats)
            else:
                updated_stats = old_stats.accum(new_stat_coeff = m / self.learn_after, new_stats = new_stats)
            #

        if do_learn_updates:
            learning_updates = self.m_step.get_updates(self, updated_stats)
        else:
            learning_updates = {}

        if do_stats_updates:
            self.suff_stat_holder.update(learning_updates, updated_stats)

        self.censor_updates(learning_updates)

        return function([X], updates = learning_updates)
    #

    def censor_updates(self, updates):

        if self.alpha in updates:
            updates[self.alpha] = T.clip(updates[self.alpha],self.min_alpha,self.max_alpha)

        if self.B in updates:
            updates[self.B] = T.clip(updates[self.B],self.min_B,self.max_B)

        if self.bias_hid in updates:
            updates[self.bias_hid] = T.clip(updates[self.bias_hid],self.min_bias_hid,self.max_bias_hid)

        if self.hard_max_step is not None:
            for param in updates:
                updates[param] = T.clip(updates[param],param-self.hard_max_step,param+self.hard_max_step)


    @classmethod
    def log_likelihood_vhs_needed_stats(cls):
        h = S3C.log_likelihood_h_needed_stats()
        s = S3C.log_likelihood_s_given_h_needed_stats()
        v = S3C.log_likelihood_v_given_hs_needed_stats()

        union = h.union(s).union(v)

        return union


    def log_likelihood_vhs(self, stats):

        log_likelihood_v_given_hs = self.log_likelihood_v_given_hs(stats)
        log_likelihood_v_given_hs = Print('log_likelihood_v_given_hs')(log_likelihood_v_given_hs)
        log_likelihood_s_given_h  = self.log_likelihood_s_given_h(stats)
        log_likelihood_s_given_h = Print('log_likelihood_s_given_h')(log_likelihood_s_given_h)
        log_likelihood_h          = self.log_likelihood_h(stats)
        log_likelihood_h = Print('log_likelihood_h')(log_likelihood_h)

        rval = log_likelihood_v_given_hs + log_likelihood_s_given_h + log_likelihood_h

        assert len(rval.type.broadcastable) == 0

        return rval

    def log_likelihood_vhsu(self, stats):

        Z_b_term = - T.nnet.softplus(self.bias_hid).sum()
        Z_alpha_term = 0.5 * T.log(self.alpha).sum()

        N = np.cast[config.floatX]( self.nhid )
        D = np.cast[config.floatX]( self.nvis )
        half = np.cast[config.floatX]( 0.5)
        one = np.cast[config.floatX](1.)
        two = np.cast[config.floatX](2.)
        four = np.cast[config.floatX](4.)
        pi = np.cast[config.floatX](np.pi)

        Z_B_term = half * (np.square(N) + one) * T.log(self.B).sum()

        Z_constant_term = - half * (N+D)*np.log(two*pi) - half * np.square(N)*D*np.log(four*pi)


        negative_log_Z = Z_b_term + Z_alpha_term + Z_B_term + Z_constant_term
        negative_log_Z.name = 'negative_log_Z'
        assert len(negative_log_Z.type.broadcastable) == 0

        u_stat_1 = stats.d['u_stat_1']

        first_term = half * T.dot(self.B, (self.W.T * u_stat_1).sum(axis=0) )

        assert len(first_term.type.broadcastable) == 0

        mean_hsv = stats.d['mean_hsv']

        second_term = T.sum(self.B *  T.sum(self.W.T * mean_hsv,axis=0), axis=0)

        assert len(second_term.type.broadcastable) == 0


        mean_sq_hs = stats.d['mean_sq_hs']
        third_term = - half * N *  T.dot(self.B, T.dot(T.sqr(self.W),mean_sq_hs))

        mean_hs = stats.d['mean_hs']

        fourth_term = T.dot(self.mu, self.alpha * mean_hs)

        mean_sq_v = stats.d['mean_sq_v']

        fifth_term = - half * T.dot(self.B, mean_sq_v)

        mean_sq_s = stats.d['mean_sq_s']

        sixth_term = - half * T.dot(self.alpha, mean_sq_s)

        mean_h = stats.d['mean_h']

        seventh_term = T.dot(self.bias_hid, mean_h)

        eighth_term = - half * T.dot(mean_h, self.alpha * T.sqr(self.mu))

        u_stat_2 = stats.d['u_stat_2']

        ninth_term = - (one / four ) * T.dot( self.B, u_stat_2)

        ne_first_quarter = first_term + second_term
        assert len(ne_first_quarter.type.broadcastable) == 0

        ne_second_quarter = third_term + fourth_term
        assert len(ne_second_quarter.type.broadcastable) ==0


        ne_first_half = ne_first_quarter + ne_second_quarter
        assert len(ne_first_half.type.broadcastable) == 0

        ne_second_half = fifth_term + sixth_term + seventh_term + eighth_term + ninth_term
        assert len(ne_second_half.type.broadcastable) == 0

        negative_energy = ne_first_half + ne_second_half
        negative_energy.name = 'negative_energy'
        assert len(negative_energy.type.broadcastable) ==0

        rval = negative_energy + negative_log_Z
        assert len(rval.type.broadcastable) == 0
        rval.name = 'log_likelihood_vhsu'

        return rval


    def log_likelihood_u_given_hs(self, stats):
        """Note: drops some constant terms """

        NH = np.cast[config.floatX](self.nhid)

        mean_sq_hs = stats.d['mean_sq_hs']
        cov_hs = stats.d['cov_hs']
        mean_D_sq_mean_Q_hs = stats.d['mean_D_sq_mean_Q_hs']

        term1 = 0.5 * T.sqr(NH) * T.sum(T.log(self.B))
        #term1 = Print('term1')(term1)
        term2 = 0.5 * (NH + 1) * T.dot(self.B,T.dot(self.W,mean_sq_hs))
        #term2 = Print('term2')(term2)
        term3 = - (self.B *  ( cov_hs.dimshuffle('x',0,1) * self.W.dimshuffle(0,1,'x') *
                        self.W.dimshuffle(0,'x',1)).sum(axis=(1,2))).sum()
        #term3 = Print('term3')(term3)
        a = T.dot(T.sqr(self.W), mean_D_sq_mean_Q_hs)
        term4 = -0.5 * T.dot(self.B, a)
        #term4 = Print('term4')(term4)

        rval = term1 + term2 + term3 + term4

        return rval

    @classmethod
    def log_likelihood_v_given_hs_needed_stats(cls):
        return set(['mean_sq_v','mean_hsv','cov_hs'])

    def log_likelihood_v_given_hs(self, stats):

        """
        E_v,h,s \sim Q log P( v | h, s)
        = E_v,h,s \sim Q log sqrt(B/2 pi) exp( - 0.5 B (v- W[v,:] (h*s) )^2)
        = E_v,h,s \sim Q 0.5 log B - 0.5 log 2 pi - 0.5 B v^2 + B W[v,:] (h*s) - 0.5 B sum_i sum_j W[v,i] W[v,j] h_i s_i h_j s_j
        = 0.5 log B - 0.5 log 2 pi - 0.5 B v^2 + B W[v,:] (h*s) - 0.5 B sum_i,j W[v,i] W[v,j] cov(h_i s_i, h_j s_j)
        """

        half = as_floatX(0.5)
        two = as_floatX(2.)
        pi = as_floatX(np.pi)
        N = as_floatX(self.nhid)

        mean_sq_v = stats.d['mean_sq_v']
        mean_hsv  = stats.d['mean_hsv']
        cov_hs = stats.d['cov_hs']

        term1 = half * T.sum(T.log(self.B))
        term2 = - half * N * T.log(two * pi)
        term3 = - half * T.dot(self.B, mean_sq_v)
        term4 = (self.B * T.dot(self.W, mean_hsv)).sum()
        term5 = - half * (self.B *  ( cov_hs.dimshuffle('x',0,1) * self.W.dimshuffle(0,1,'x') *
                        self.W.dimshuffle(0,'x',1)).sum(axis=(1,2))).sum(axis=0)

        rval = term1 + term2 + term3 + term4 + term5

        assert len(rval.type.broadcastable) == 0

        return rval

    @classmethod
    def log_likelihood_s_given_h_needed_stats(cls):
        return set(['mean_h','mean_hs','mean_sq_s'])

    def log_likelihood_s_given_h(self, stats):

        """
        E_h,s\sim Q log P(s|h)
        = E_h,s\sim Q log sqrt( alpha / 2pi) exp(- 0.5 alpha (s-mu h)^2)
        = E_h,s\sim Q log sqrt( alpha / 2pi) - 0.5 alpha (s-mu h)^2
        = E_h,s\sim Q  0.5 log alpha - 0.5 log 2 pi - 0.5 alpha s^2 + alpha s mu h + 0.5 alpha mu^2 h^2
        = E_h,s\sim Q 0.5 log alpha - 0.5 log 2 pi - 0.5 alpha s^2 + alpha mu h s + 0.5 alpha mu^2 h
        = 0.5 log alpha - 0.5 log 2 pi - 0.5 alpha mean_sq_s + alpha mu mean_hs - 0.5 alpha mu^2 mean_h
        """

        mean_h = stats.d['mean_h']
        mean_sq_s = stats.d['mean_sq_s']
        mean_hs = stats.d['mean_hs']

        half = as_floatX(0.5)
        two = as_floatX(2.)
        N = as_floatX(self.nhid)
        pi = as_floatX(np.pi)

        term1 = half * T.log( self.alpha ).sum()
        term2 = - half * N * T.log(two*pi)
        term3 = - half * T.dot( self.alpha , mean_sq_s )
        term4 = T.dot(self.mu*self.alpha,mean_hs)
        term5 = - half * T.dot(T.sqr(self.mu), self.alpha * mean_h)

        rval = term1 + term2 + term3 + term4 + term5

        assert len(rval.type.broadcastable) == 0

        return rval

    @classmethod
    def log_likelihood_h_needed_stats(cls):
        return set(['mean_h'])

    def log_likelihood_h(self, stats):
        """ Returns the expected log probability of the vector h
            under the model when the data is drawn according to
            stats
        """

        """
            E_h\sim Q log P(h)
            = E_h\sim Q log exp( bh) / (1+exp(b))
            = E_h\sim Q bh - softplus(b)
        """

        mean_h = stats.d['mean_h']

        term1 = T.dot(self.bias_hid, mean_h)
        term2 = - T.nnet.softplus(self.bias_hid).sum()

        rval = term1 + term2

        assert len(rval.type.broadcastable) == 0

        return rval


    def redo_theano(self):
        init_names = dir(self)

        self.w = T.dot(self.B, T.sqr(self.W))

        X = T.matrix()
        X.tag.test_value = np.cast[config.floatX](self.rng.randn(5,self.nvis))

        if self.learn_after is not None:
            self.learn_func = self.make_learn_func(X, learn = True )
            self.accum_func = self.make_learn_func(X, learn = False )
        else:
            self.learn_func = self.make_learn_func(X)
        #

        final_names = dir(self)

        self.register_names_to_del([name for name in final_names if name not in init_names])
    #

    def learn(self, dataset, batch_size):
        self.learn_mini_batch(dataset.get_batch_design(batch_size))
    #


    def learn_mini_batch(self, X):

        if self.learn_after is not None:
            if self.monitor.examples_seen >= self.learn_after:
                self.learn_func(X)
            else:
                self.accum_func(X)
        else:
            self.learn_func(X)

        """cov_hs = self.suff_stat_holder.d['cov_hs'].get_value(borrow=True)
        a,b = np.linalg.eigh(cov_hs)

        assert not np.any(np.isnan(a))
        assert not np.any(np.isinf(a))
        print 'minimum eigenvalue: '+str(a.min())
        assert a.min() >= 0"""

        if self.monitor.examples_seen % 10000 == 0:
            print ""
            b = self.bias_hid.get_value(borrow=True)
            p = 1./(1.+np.exp(-b))
            print 'p: ',(p.min(),p.mean(),p.max())
            B = self.B.get_value(borrow=True)
            print 'B: ',(B.min(),B.mean(),B.max())
            mu = self.mu.get_value(borrow=True)
            print 'mu: ',(mu.min(),mu.mean(),mu.max())
            alpha = self.alpha.get_value(borrow=True)
            print 'alpha: ',(alpha.min(),alpha.mean(),alpha.max())
            W = self.W.get_value(borrow=True)
            print 'W: ',(W.min(),W.mean(),W.max())
    #

    def get_weights_format(self):
        return ['v','h']

class E_step(object):
    def __init__(self):
        self.model = None

    def register_model(self, model):
        self.model = model

    def mean_field(self, V):
        raise NotImplementedError()

class VHS_E_Step(E_step):
    """ A variational E_step that works by running damped mean field
        on the original  model

        All variables are updated simultaneously, in parallel. The
        spike variables are updated with a fixed damping. The slab
        variables are updated with a unit-specific damping designed
        to ensure stability.

        Note that while the updates for s_i and h_i have common
        sub-expressions, the update for h_i does not depend on s_i
        and vice versa. Thus the question of whether to update
        h and s in parallel (as opposed to updating h, then updating
        s, or vice versa) is probably not that important; the only
        difference is how quickly changes in one variable will
        appear in the interaction term of the other variable.

        """

    def __init__(self, h_new_coeff_schedule):
        """Parameters
        --------------
        h_new_coeff_schedule:
            list of coefficients to put on the new value of h on each damped mean field step
                    (coefficients on s are driven by a special formula)
            length of this list determines the number of mean field steps
        """
        self.h_new_coeff_schedule = h_new_coeff_schedule

        super(VHS_E_Step, self).__init__()

    def init_mf_H(self, V):
        #just use the prior
        return T.nnet.sigmoid(self.model.bias_hid).dimshuffle('x',0)

    def init_mf_Mu1(self, V):
        #just use the prior
        return self.model.mu.dimshuffle('x',0)



    def mean_field_A(self, V, H, Mu1):

        mu = self.model.mu
        alpha = self.model.alpha
        W = self.model.W
        B = self.model.B
        w = self.model.w

        BW = B.dimshuffle(0,'x') * W

        HS = H * Mu1

        mean_term = mu * alpha

        data_term = T.dot(V, BW)

        iterm_part_1 = - T.dot(T.dot(HS, W.T), BW)
        iterm_part_2 = w * HS

        interaction_term = iterm_part_1 + iterm_part_2

        A = mean_term + data_term + interaction_term

        return A

    def mean_field_Mu1(self, A):

        alpha = self.model.alpha
        w = self.model.w

        denom = alpha + w

        Mu1 =  A / denom

        return Mu1

    def mean_field_Sigma1(self):
        #TODO: this is a bad name, since it's really more like sigma^2

        return 1./ (self.model.alpha + self.model.w )

    def mean_field_H(self, A):

        half = as_floatX(.5)

        term1 = half * T.sqr(A)

        term2 = self.model.bias_hid

        term3 = - T.sqr(self.model.mu) * self.model.alpha

        term4 = -half * T.log(self.model.alpha + self.model.w)

        term5 = half * T.log(self.model.alpha)

        arg_to_sigmoid = term1 + term2 + term3 + term4 + term5

        H = T.nnet.sigmoid(arg_to_sigmoid)

        return H

    def damp_H(self, H, new_H, new_coeff):
        return new_coeff * new_H + (1. - new_coeff) * H

    def damp_Mu1(self, Mu1, new_Mu1):
        rho = 0.5
        ceiling = 1000.

        positives = Mu1 > 0
        non_positives = 1. - positives
        negatives = Mu1 < 0
        non_negatives = 1. - negatives

        rval = T.clip(new_Mu1, - rho * positives * Mu1 - non_positives * ceiling, non_negatives * ceiling - rho * negatives * Mu1 )

        return rval

    def mean_field(self, V):
        alpha = self.model.alpha

        sigma0 = 1. / alpha
        mu0 = T.zeros_like(sigma0)

        H   =    self.init_mf_H(V)
        Mu1 =    self.init_mf_Mu1(V)

        for new_coeff in self.h_new_coeff_schedule:
            A = self.mean_field_A(V = V, H = H, Mu1 = Mu1)
            new_Mu1 = self.mean_field_Mu1(A = A)
            new_H = self.mean_field_H(A = A)

            H = self.damp_H(H = H, new_H = new_H, new_coeff = new_coeff)
            Mu1 = self.damp_Mu1(Mu1 = Mu1, new_Mu1 = new_Mu1)


        Sigma1 = self.mean_field_Sigma1()

        return {
                'H' : H,
                'mu0' : mu0,
                'Mu1' : Mu1,
                'sigma0' : sigma0,
                'Sigma1': Sigma1,
                }


class VHSU_E_Step(E_step):
    """ A variational E-step that works by running mean field on
        the auxiliary variable model """

    def __init__(self, N_schedule):
        """
        parameters:
            N_schedule: list of values to use for N throughout mean field updates.
                    len(N_schedule) determines # mean field steps
        """
        self.N_schedule = N_schedule

        super(VHSU_E_Step, self).__init__()


    def init_mf_Mu1(self, V):
        #Mu1 = (self.alpha*self.mu + T.dot(V*self.B,self.W))/(self.alpha+self.w)
        #Just use the prior
        Mu1 = self.model.mu.dimshuffle('x',0)

        Mu1.name = "init_mf_Mu1"

        return Mu1
    #


    def mean_field_H(self, U, V, NH):

        BW = self.model.W * (self.model.B.dimshuffle(0,'x'))

        filt = T.dot(V,BW)

        u_contrib = (U * BW.dimshuffle('x',1,0)).sum(axis=2)

        pre_sq = filt - u_contrib + self.model.alpha * self.model.mu

        sq_term = T.sqr(pre_sq)

        beta = self.model.alpha + NH * self.model.w

        log_term = T.log(1.0 + NH * self.model.w / self.model.alpha )

        H = T.nnet.sigmoid(-self.h_coeff() + 0.5 * sq_term / beta  - 0.5 * log_term )

        H.name = "mean_field_H"

        return H
    #

    def mean_field_Mu1(self, U, V, NH):

        beta = self.model.alpha + NH * self.model.w

        BW = self.model.W * self.model.B.dimshuffle(0,'x')

        filt = T.dot(V,BW)

        u_mod = - (U * BW.dimshuffle('x',1,0)).sum(axis=2)

        Mu1 = (filt + u_mod + self.model.alpha * self.model.mu) / beta

        Mu1.name = "mean_field_Mu1"

        return Mu1
    #


    def mean_field_Sigma1(self, NH):
        Sigma1 = 1./(self.model.alpha + NH * self.model.w)

        Sigma1.name = "mean_field_Sigma1"

        return Sigma1
    #


    def mean_field(self, V):
        alpha = self.model.alpha

        sigma0 = 1. / alpha
        mu0 = T.zeros_like(sigma0)

        H   =    self.init_mf_H(V)
        Mu1 =    self.init_mf_Mu1(V)


        for NH in self.N_schedule:
            U   = self.mean_field_U  (H = H, Mu1 = Mu1, NH = NH)
            H   = self.mean_field_H  (U = U, V = V,     NH = NH)
            Mu1 = self.mean_field_Mu1(U = U, V = V,     NH = NH)


        Sigma1 = self.mean_field_Sigma1(NH = np.cast[config.floatX](self.model.nhid))

        return {
                'H' : H,
                'mu0' : mu0,
                'Mu1' : Mu1,
                'sigma0' : sigma0,
                'Sigma1': Sigma1,
                'U' : U
                }
    #

    def mean_field_U(self, H, Mu1, NH):

        W = self.model.W

        prod = Mu1 * H

        first_term = T.dot(prod, W.T)
        first_term_broadcast = first_term.dimshuffle(0,'x',1)

        W_broadcast = W.dimshuffle('x',1,0)
        prod_broadcast = prod.dimshuffle(0,1,'x')

        second_term = NH * W_broadcast * prod_broadcast

        U = first_term_broadcast - second_term

        U.name = "mean_field_U"

        return U
    #

    def h_coeff(self):
        """ Returns the coefficient on h in the energy function """
        return - self.model.bias_hid  + 0.5 * T.sqr(self.model.mu) * self.model.alpha

    def init_mf_H(self,V):
        nhid = self.model.nhid
        w = self.model.w
        alpha = self.model.alpha
        mu = self.model.mu
        W = self.model.W
        B = self.model.B

        NH = np.cast[config.floatX] ( nhid)
        arg_to_log = 1.+(1./alpha) * NH * w

        hid_vec = alpha * mu
        #assert (hasattr(V,'__array__') or (V.tag.test_value is not None))
        dotty_thing = T.dot(V*B, W)
        pre_sq = hid_vec + dotty_thing
        numer = T.sqr(pre_sq)
        denom = alpha + w
        frac = numer/ denom

        first_term = 0.5 *  frac

        H = T.nnet.sigmoid( first_term - self.h_coeff() - 0.5 * T.log(arg_to_log) )


        #just use the prior
        H = T.nnet.sigmoid( self.model.bias_hid )

        #H = Print('init_mf_H')(H)

        return H
    #


class M_Step(object):

    def needed_stats(self):
        """ Return a set of string names of the sufficient statistics that will be needed
            TODO: do this automatically instead of requiring it to be hard-coded """
        raise NotImplementedError()

    def get_updates(self, model, stats):
        raise NotImplementedError()

    def get_monitoring_channels(self, V, model):
        return {}

class VHS_M_Step(M_Step):
    """ An M-step based on learning using the distribution
        over V,H, and S.

        In conjunction with VHS_E_Step this does variational
        EM in the original model. (Haven't run this yet as of
        time of writing this comment)

        In conjunction with VHSU_E_Step: we have no theoretical
        justification for this. In experiments on CIFAR it learns
        a mixture of gabors and dead filters.
    """

    def get_monitoring_channels(self, V, model):

        hid_observations = model.e_step.mean_field(V)

        stats = SufficientStatistics.from_observations(needed_stats = S3C.log_likelihood_vhs_needed_stats(),
                X = V, **hid_observations)

        obj = model.log_likelihood_vhs(stats)

        return { 'log_likelihood_vhs' : obj }

class VHSU_M_Step(M_Step):
    """ An M-step based on learning using the distribution over
        V,H,S, and U-- i.e. good old-fashioned, theoretically
        justified EM

        This M step has been unit tested and seems to work correctly
        in unit tests. It has not been shown to work well in learning
        experiments. That could mean the auxiliary variables are a bad
        idea or it could mean something is wrong with the VHSU E step.
    """

    def get_monitoring_channels(self, V, model):

        hidden_obs  = model.e_step.mean_field(V)

        stats = SufficientStatistics.from_observations(needed_stats = S3C.log_likelihood_vhsu_needed_stats(), X =V, \
                                                            N = np.cast[config.floatX](model.nhid),
                                                            B = model.B,
                                                            W = model.W,
                                                            **hidden_obs)

        obj = model.log_likelihood_vhsu(stats)

        return { 'log_likelihood_vhsu' : obj }


def take_step(model, W, bias_hid, alpha, mu, B, new_coeff):
    """
    Returns a dictionary of learning updates of the form
        model.param := new_coeff * param + (1-new_coeff) * model.param
    """

    new_coeff = np.cast[config.floatX](new_coeff)

    def step(old, new):
        if new_coeff == 1.0:
            return new
        else:
            rval =  new_coeff * new + (np.cast[config.floatX](1.)-new_coeff) * old

        assert rval.dtype == config.floatX

        return rval

    learning_updates = \
        {
            model.W: step(model.W, W),
            model.bias_hid: step(model.bias_hid,bias_hid),
            model.alpha: step(model.alpha, alpha),
            model.mu: step(model.mu, mu),
            model.B: step(model.B, B)
        }

    return learning_updates

class VHS_Solve_M_Step(VHS_M_Step):

    def __init__(self, new_coeff):
        self.new_coeff = np.cast[config.floatX](float(new_coeff))

    def needed_stats(self):
        return S3C.solve_vhs_needed_stats()

    def get_updates(self, model, stats):

        W, bias_hid, alpha, mu, B = model.solve_vhs_from_stats(stats)

        learning_updates = take_step(model, W, bias_hid, alpha, mu, B, self.new_coeff)

        return learning_updates

class VHSU_Solve_M_Step(VHSU_M_Step):

    def __init__(self, new_coeff):
        self.new_coeff = np.cast[config.floatX](float(new_coeff))

    def get_updates(self, model, stats):

        W, bias_hid, alpha, mu, B = model.solve_vhsu_from_stats(stats)

        learning_updates = take_step(model, W, bias_hid, alpha, mu, B, self.new_coeff)

        return learning_updates

class VHS_Grad_M_Step(VHS_M_Step):

    def __init__(self, learning_rate):
        self.learning_rate = np.cast[config.floatX](float(learning_rate))

    def get_updates(self, model, stats):

        params = model.get_params()

        obj = model.log_likelihood_vhs(stats)

        grads = T.grad(obj, params, consider_constant = stats.d.values())

        updates = {}

        for param, grad in zip(params, grads):
            updates[param] = param + self.learning_rate * grad

        return updates

    def needed_stats(self):
        return S3C.log_likelihood_vhs_needed_stats()

class VHSU_Grad_M_Step(VHSU_M_Step):

    def __init__(self, learning_rate):
        self.learning_rate = np.cast[config.floatX](float(learning_rate))

    def get_updates(self, model, stats):

        params = model.get_params()

        obj = model.log_likelihood_vhsu(stats)

        grads = T.grad(obj, params, consider_constant = stats.d.values())

        updates = {}

        for param, grad in zip(params, grads):
            #if param is model.W:
            #    grad = Print('grad_W',attrs=['min','mean','max'])(grad)

            updates[param] = param + self.learning_rate * grad

        return updates



