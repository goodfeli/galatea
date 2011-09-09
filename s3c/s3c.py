from pylearn2.models.model import Model
from theano import config, function, shared
import theano.tensor as T
import numpy as np
from theano.sandbox.linalg.ops import alloc_diag
#from theano.sandbox.linalg.ops import extract_diag
from theano.sandbox.linalg.ops import matrix_inverse
import warnings
from theano.printing import Print
from pylearn2.utils import make_name, sharedX, as_floatX
from pylearn2.monitor import Monitor
#import copy
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
                    "second_hs"                :   sharedX(np.zeros((nhid,nhid)), 'second_hs'),
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

        H_name = make_name(H, 'anon_H')
        Mu1_name = make_name(Mu1, 'anon_Mu1')

        #mean_h
        assert H.dtype == config.floatX
        mean_h = T.mean(H, axis=0)
        assert H.dtype == mean_h.dtype
        assert mean_h.dtype == config.floatX
        mean_h.name = 'mean_h('+H_name+')'

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
        mean_hs.name = 'mean_hs(%s,%s)' % (H_name, Mu1_name)
        mean_D_sq_mean_Q_hs = T.mean(T.sqr(mean_HS), axis=0)

        #mean_sq_hs
        mean_sq_HS = H * (Sigma1 + T.sqr(Mu1))
        mean_sq_hs = T.mean(mean_sq_HS, axis=0)
        mean_sq_hs.name = 'mean_sq_hs(%s,%s)' % (H_name, Mu1_name)

        #second_hs
        outer_prod = T.dot(mean_HS.T,mean_HS)
        outer_prod.name = 'outer_prod<from_observations>'
        outer = outer_prod/m
        mask = T.identity_like(outer)
        second_hs = (1.-mask) * outer + alloc_diag(mean_sq_hs)
        second_hs.name = 'exp_outer_hs(%s,%s)' % (H_name, Mu1_name)

        #mean_hsv
        sum_hsv = T.dot(mean_HS.T,X)
        sum_hsv.name = 'sum_hsv<from_observations>'
        mean_hsv = sum_hsv / m

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
            dotA = T.dot(T.sqr(mean_HS),T.sqr(W.T))
            dotA.name = 'dotA'
            term2 = two * N * dotA
            #term2 = Print('us2 term2',attrs=['mean'])(term2)
            dotB = T.dot(mean_HS, W.T)
            dotB.name = 'dotB'
            term3 = - two * T.sqr( dotB )
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
                    "second_hs"             :   second_hs,
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


class DebugEnergy:
    def __init__(self,
                    h_term = True,
                    s_term_1 = True,
                    s_term_2 = True,
                    s_term_3 = True,
                    v_term = True):
        self.h_term = h_term
        self.s_term_1 = s_term_1
        self.s_term_2 = s_term_2
        self.s_term_3 = s_term_3
        self.v_term = v_term

        for field in dir(self):
            if type(field) == type(True) and not field:
                print "HACK: some terms of energy / expected energy zeroed out"
                break


class S3C(Model):
    def __init__(self, nvis, nhid, irange, init_bias_hid,
                       init_B, min_B, max_B,
                       init_alpha, min_alpha, max_alpha, init_mu,
                       new_stat_coeff,
                       e_step,
                       m_step,
                       W_eps = 1e-6, mu_eps = 1e-8, b_eps = 0.,
                        min_bias_hid = -1e30,
                        max_bias_hid = 1e30,
                        min_mu = -1e30,
                        max_mu = 1e30,
                        tied_B = False,
                       learn_after = None, hard_max_step = None,
                       monitor_stats = None,
                       monitor_functional = False,
                       recycle_q = 0,
                       seed = None,
                       disable_W_update = False):
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
        mu_eps:      L2 regularization parameter for linear regression problem for mu
        b_eps:       L2 regularization parameter for linear regression problem for b
        learn_after: only applicable when new_stat_coeff < 1.0
                        begins learning parameters and decaying sufficient statistics
                        after seeing learn_after examples
                        until this time, only accumulates sufficient statistics
        hard_max_step:  if set to None, has no effect
                        otherwise, every element of every parameter is not allowed to change
                        by more than this amount on each M-step. This is basically a hack
                        introduced to prevent explosion in gradient descent.
        tied_B:         if True, use a scalar times identity for the precision on visible units.
                        otherwise use a diagonal matrix for the precision on visible units
        monitor_stats:  a list of sufficient statistics to monitor on the monitoring dataset
        monitor_functional: if true, monitors the EM functional on the monitoring dataset
        recycle_q: if nonzero, initializes the e-step with the output of the previous iteration's
                    e-step. obviously this should only be used if you are using the same data
                    in each batch. when recycle_q is nonzero, it should be set to the batch size.
        disable_W_update: if true, doesn't update W (for debugging)
        """

        super(S3C,self).__init__()

        if monitor_stats is None:
            self.monitor_stats = []
        else:
            self.monitor_stats = [ elem for elem in monitor_stats ]

        self.seed = seed

        self.disable_W_update = disable_W_update
        self.monitor_functional = monitor_functional
        self.W_eps = np.cast[config.floatX](float(W_eps))
        self.mu_eps = np.cast[config.floatX](float(mu_eps))
        self.b_eps = np.cast[config.floatX](float(b_eps))
        self.nvis = nvis
        self.nhid = nhid
        self.irange = irange
        self.init_bias_hid = init_bias_hid
        self.init_alpha = float(init_alpha)
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
        self.init_B = float(init_B)
        self.min_B = float(min_B)
        self.max_B = float(max_B)
        self.e_step = e_step
        self.e_step.register_model(self)
        self.m_step = m_step
        self.init_mu = init_mu
        self.min_mu = min_mu
        self.max_mu = max_mu
        self.min_bias_hid = min_bias_hid
        self.max_bias_hid = max_bias_hid
        self.recycle_q = recycle_q

        self.tied_B = tied_B

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
        if self.seed is None:
            self.rng = np.random.RandomState([1.,2.,3.])
        else:
            self.rng = np.random.RandomState(self.seed)

    def redo_everything(self):
        self.W = sharedX(self.rng.uniform(-self.irange, self.irange, (self.nvis, self.nhid)), name = 'W')
        self.bias_hid = sharedX(np.zeros(self.nhid)+self.init_bias_hid, name='bias_hid')
        self.alpha = sharedX(np.zeros(self.nhid)+self.init_alpha, name = 'alpha')
        self.mu = sharedX(np.zeros(self.nhid)+self.init_mu, name='mu')
        if self.tied_B:
            self.B_driver = sharedX(0.0+self.init_B, name='B')
        else:
            self.B_driver = sharedX(np.zeros(self.nvis)+self.init_B, name='B')


        if self.new_stat_coeff < 1.0:
            self.suff_stat_holder = SufficientStatisticsHolder(nvis = self.nvis, nhid = self.nhid,
                    needed_stats = self.m_step.needed_stats() )

        self.test_batch_size = 5

        if self.recycle_q:
            self.prev_H = sharedX(np.zeros((self.test_batch_size,self.nhid)), name="prev_H")
            self.prev_Mu1 = sharedX(np.zeros((self.test_batch_size,self.nhid)), name="prev_Mu1")

        self.debug_m_step = False
        if self.debug_m_step:
            self.em_functional_diff = sharedX(0.)

        self.redo_theano()


        if self.recycle_q:
            self.prev_H.set_value( np.cast[self.prev_H.dtype]( np.zeros((self.recycle_q, self.nhid)) + 1./(1.+np.exp(-self.bias_hid.get_value()))))
            self.prev_Mu1.set_value( np.cast[self.prev_Mu1.dtype]( np.zeros((self.recycle_q, self.nhid)) + self.mu.get_value() ) )

    def em_functional(self, H, sigma0, Sigma1, stats):
        """ Returns the em_functional for a single batch of data
            stats is assumed to be computed from and only from
            the same data points that yielded H """

        assert self.new_stat_coeff == 1.0

        entropy_term = (self.entropy_hs(H = H, sigma0 = sigma0, Sigma1 = Sigma1)).mean()
        likelihood_term = self.expected_log_prob_vhs(stats)

        em_functional = likelihood_term + entropy_term

        return em_functional

    def get_monitoring_channels(self, V):


        rval = self.m_step.get_monitoring_channels(V, self)

        from_e_step = self.e_step.get_monitoring_channels(V, self)

        rval.update(from_e_step)

        monitor_stats = len(self.monitor_stats) > 0

        if monitor_stats or self.monitor_functional:

            obs = self.e_step.mean_field(V)

            needed_stats = set(self.monitor_stats)

            if self.monitor_functional:
                needed_stats = needed_stats.union(S3C.expected_log_prob_vhs_needed_stats())

            stats = SufficientStatistics.from_observations( needed_stats = needed_stats,
                                                            X = V, ** obs )

            H = obs['H']
            sigma0 = obs['sigma0']
            Sigma1 = obs['Sigma1']

            if self.monitor_functional:
                em_functional = self.em_functional(H = H, sigma0 = sigma0, Sigma1 = Sigma1, stats = stats)

                rval['em_functional'] = em_functional


            if monitor_stats:



                for stat in self.monitor_stats:
                    stat_val = stats.d[stat]

                    rval[stat+'_min'] = T.min(stat_val)
                    rval[stat+'_mean'] = T.mean(stat_val)
                    rval[stat+'_max'] = T.max(stat_val)

        return rval

    def get_params(self):
        return [self.W, self.bias_hid, self.alpha, self.mu, self.B_driver ]

    @classmethod
    def solve_vhs_needed_stats(cls):
        return set([ 'second_hs',
                 'mean_hsv',
                 'mean_v',
                 'mean_sq_v',
                 'mean_sq_s',
                 'mean_sq_hs',
                 'mean_hs',
                 'mean_h'
                ])

    def solve_vhs_from_stats(self, stats):

        """
            One thing that's complicated about debugging
            this method is that some of the updates depend
            on the other updates. For example, the new
            value of B depends on the new value of W.
            Because these updates are later passed to
            censor_updates, some of the values computed
            here may not be correct.
            In here, we don't update W if disable_W_update
            is set to True, but we haven't compensated for
            other parameter dependencies.
        """

        """Solve multiple linear regression problem where
         W is a matrix used to predict v from h*s


        second_hs[i,j] = E_D,Q h_i s_i h_j s_j   (note that diagonal has different formula)
        """

        second_hs = stats.d['second_hs']
        assert second_hs.dtype == config.floatX
        #mean_hsv[i,j] = E_D,Q h_i s_i v_j
        mean_hsv = stats.d['mean_hsv']

        regularized = second_hs + alloc_diag(T.ones_like(self.mu) * self.W_eps)
        assert regularized.dtype == config.floatX


        inv = matrix_inverse(regularized)
        assert inv.dtype == config.floatX

        inv_prod = T.dot(inv,mean_hsv)
        inv_prod.name = 'inv_prod'
        new_W = inv_prod.T
        assert new_W.dtype == config.floatX

        if self.disable_W_update:
            new_W = self.W

        #Solve for B by setting gradient of log likelihood to 0
        mean_sq_v = stats.d['mean_sq_v']

        one = as_floatX(1.)
        two = as_floatX(2.)

        denom1 = mean_sq_v

        denom2 = - two * (new_W * mean_hsv.T).sum(axis=1)
        denom3 = (second_hs.dimshuffle('x',0,1)*new_W.dimshuffle(0,1,'x')*new_W.dimshuffle(0,'x',1)).sum(axis=(1,2))

        denom = denom1 + denom2 + denom3

        #denom = T.clip(denom1 + denom2 + denom3, 1e-10, 1e8)

        new_B = one / denom

        if self.tied_B:
            new_B = new_B.mean()

        mean_hs = stats.d['mean_hs']

        # Now a linear regression problem where mu_i is used to predict
        # s_i from h_i

        # mu_i = ( h^T h + reg)^-1 h^T s_i

        mean_h = stats.d['mean_h']
        assert mean_h.dtype == config.floatX
        reg = self.mu_eps
        new_mu = mean_hs/(mean_h+reg)


        mean_sq_s = stats.d['mean_sq_s']
        mean_sq_hs = stats.d['mean_h']

        s_denom1 = mean_sq_s
        s_denom2 = - two * new_mu * mean_hs
        s_denom3 = T.sqr(new_mu) * mean_h


        s_denom = s_denom1 + s_denom2 + s_denom3

        new_alpha = one / s_denom


        #probability of hiddens just comes from sample counting
        #to put it back in bias_hid space apply sigmoid inverse

        p = T.clip(mean_h,np.cast[config.floatX](1e-8),np.cast[config.floatX](1.-1e-8))
        p.name = 'mean_h_clipped'

        assert p.dtype == config.floatX

        bias_hid = T.log( - p / (p-1.+self.b_eps) )

        assert bias_hid.dtype == config.floatX

        return new_W, bias_hid, new_alpha, new_mu, new_B

    @classmethod
    def solve_vhsu_needed_stats(cls):
        return set(['mean_hsv',
                    'mean_sq_s',
                    'u_stat_1',
                    'mean_sq_hs',
                    'mean_hs',
                    'mean_h',
                    'u_stat_2',
                    'mean_sq_v'])

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

        dotC =  T.dot(T.sqr(new_W), mean_sq_hs)
        dotC.name = 'dotC'
        denom1 = N * dotC
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


    def energy_vhs(self, V, H, S, debug_energy = None):
        " H MUST be binary "

        if debug_energy is None:
            debug_energy = DebugEnergy()



        h_term = - T.dot(H, self.bias_hid)
        assert len(h_term.type.broadcastable) == 1

        if not debug_energy.h_term:
            h_term = 0.


        s_term_1 = T.dot(T.sqr(S), self.alpha)/2.
        s_term_2 = -T.dot(S * self.mu * H , self.alpha)
        #s_term_3 = T.dot(T.sqr(self.mu * H), self.alpha)/2.
        s_term_3 = T.dot(T.sqr(self.mu) * H, self.alpha) / 2.

        if not debug_energy.s_term_1:
            s_term_1 = 0.

        if not debug_energy.s_term_2:
            s_term_2 = 0.

        if not debug_energy.s_term_3:
            s_term_3 = 0.

        s_term = s_term_1 + s_term_2 + s_term_3
        #s_term = T.dot( T.sqr( S - self.mu * H) , self.alpha) / 2.
        assert len(s_term.type.broadcastable) == 1


        recons = T.dot(H*S, self.W.T)

        v_term_1 = T.dot( T.sqr(V), self.B) / 2.
        v_term_2 = T.dot( - V * recons, self.B)
        v_term_3 = T.dot( T.sqr(recons), self.B) / 2.

        v_term = v_term_1 + v_term_2 + v_term_3

        #v_term = T.dot( T.sqr( V - recons), self. B) / 2.
        assert len(v_term.type.broadcastable) == 1


        if not debug_energy.v_term:
            v_term = 0.

        rval = h_term + s_term + v_term
        assert len(rval.type.broadcastable) == 1

        return rval


    def expected_energy_vhs(self, V, H, mu0, Mu1, sigma0, Sigma1, debug_energy = None):

        var_HS = H * Sigma1 + (1.-H) * sigma0

        if debug_energy is None:
            debug_energy = DebugEnergy()

        half = as_floatX(.5)

        HS = H * Mu1


        sq_HS = H * ( Sigma1 + T.sqr(Mu1))

        sq_S = sq_HS + (1.-H)*(sigma0 + T.sqr(mu0))

        presign = T.dot(H, self.bias_hid)
        presign.name = 'presign'
        h_term = - presign
        assert len(h_term.type.broadcastable) == 1

        if not debug_energy.h_term:
            h_term = 0.

        precoeff =  T.dot(sq_S, self.alpha)
        precoeff.name = 'precoeff'
        s_term_1 = half * precoeff
        assert len(s_term_1.type.broadcastable) == 1

        if not debug_energy.s_term_1:
            s_term_1 = 0.

        presign2 = T.dot(HS, self.alpha * self.mu)
        presign2.name = 'presign2'
        s_term_2 = - presign2
        assert len(s_term_2.type.broadcastable) == 1

        if not debug_energy.s_term_2:
            s_term_2 = 0.

        s_term_3 = half * T.dot(H, T.sqr(self.mu) * self.alpha)
        assert len(s_term_3.type.broadcastable) == 1

        if not debug_energy.s_term_3:
            s_term_3 = 0.

        s_term = s_term_1 + s_term_2 + s_term_3

        v_term_1 = half * T.dot(T.sqr(V),self.B)
        assert len(v_term_1.type.broadcastable) == 1

        term6_factor1 = V * self.B
        term6_factor2 = T.dot(HS, self.W.T)
        v_term_2 = - (term6_factor1 * term6_factor2).sum(axis=1)
        assert len(v_term_2.type.broadcastable) == 1

        term7_subterm1 = T.dot(T.sqr(T.dot(HS, self.W.T)), self.B)
        assert len(term7_subterm1.type.broadcastable) == 1
        #term7_subterm2 = T.dot(var_HS, self.w)
        term7_subterm2 = - T.dot( T.dot(T.sqr(HS), T.sqr(self.W.T)), self.B)
        term7_subterm3 = T.dot( T.dot(sq_HS, T.sqr(self.W.T)), self.B )

        #v_term_3 = half * (term7_subterm1 + term7_subterm2)
        v_term_3 = half * (term7_subterm1 + term7_subterm2 + term7_subterm3)
        assert len(v_term_3.type.broadcastable) == 1

        v_term = v_term_1 + v_term_2 + v_term_3

        if not debug_energy.v_term:
            v_term = 0.0

        rval = h_term + s_term + v_term

        return rval


    def entropy_h(self, H):

        #TODO: replace with actually evaluating 0 log 0 as 0
        #note: can't do 1e-8, 1.-1e-8 rounds to 1.0 in float32
        H = T.clip(H, 1e-7, 1.-1e-7)

        #H = Print('entropy_h',attrs=['min','max'])(H)

        logH = T.log(H)

        #logH = Print('logH',attrs=['min','max'])(logH)

        logOneMinusH = T.log(1.-H)

        #logOneMinusH = Print('logOneMinusH',attrs=['min','max'])(logOneMinusH)

        term1 = - T.sum( H * logH , axis=1)
        assert len(term1.type.broadcastable) == 1

        term2 = - T.sum( (1.-H) * logOneMinusH , axis =1 )
        assert len(term2.type.broadcastable) == 1

        rval = term1 + term2

        return rval

    def entropy_hs(self, H, sigma0, Sigma1):

        half = as_floatX(.5)

        one = as_floatX(1.)

        two = as_floatX(2.)

        pi = as_floatX(np.pi)

        term1_plus_term2 = self.entropy_h(H)
        assert len(term1_plus_term2.type.broadcastable) == 1

        #TODO: change Sigma1 back into a vector
        #TODO: pick new name for Sigma1; does capitalization mean it's a covariance matrix rather than a scalar
        #                               or does it mean it's a minibatch rather than one example?
        #
        term3 = T.sum( H * ( half * (T.log(Sigma1) +  T.log(two*pi) + one )  ) , axis= 1)
        assert len(term3.type.broadcastable) == 1

        term4 = T.dot( 1.-H, half * (T.log(sigma0) +  T.log(two*pi) + one ))
        assert len(term4.type.broadcastable) == 1

        rval = term1_plus_term2 + term3 + term4

        return rval


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

        if self.recycle_q:
            learning_updates[self.prev_H] = hidden_obs['H']
            learning_updates[self.prev_Mu1] = hidden_obs['Mu1']

        self.censor_updates(learning_updates)

        if self.debug_m_step:
            em_functional_before = self.em_functional(H = hidden_obs['H'],
                                                      sigma0 = hidden_obs['sigma0'],
                                                      Sigma1 = hidden_obs['Sigma1'],
                                                      stats = updated_stats)

            tmp_bias_hid = self.bias_hid
            tmp_mu = self.mu
            tmp_alpha = self.alpha
            tmp_W = self.W
            tmp_B_driver = self.B_driver

            self.bias_hid = learning_updates[self.bias_hid]
            self.mu = learning_updates[self.mu]
            self.alpha = learning_updates[self.alpha]
            if self.W in learning_updates:
                self.W = learning_updates[self.W]
            self.B_driver = learning_updates[self.B_driver]
            self.make_B_and_w()

            try:
                em_functional_after  = self.em_functional(H = hidden_obs['H'],
                                                          sigma0 = hidden_obs['sigma0'],
                                                          Sigma1 = hidden_obs['Sigma1'],
                                                          stats = updated_stats)
            finally:
                self.bias_hid = tmp_bias_hid
                self.mu = tmp_mu
                self.alpha = tmp_alpha
                self.W = tmp_W
                self.B_driver = tmp_B_driver
                self.make_B_and_w()

            em_functional_diff = em_functional_after - em_functional_before

            learning_updates[self.em_functional_diff] = em_functional_diff

        return function([X], updates = learning_updates)
    #

    def censor_updates(self, updates):

        if self.disable_W_update and self.W in updates:
            del updates[self.W]

        if self.alpha in updates:
            updates[self.alpha] = T.clip(updates[self.alpha],self.min_alpha,self.max_alpha)

        if self.mu in updates:
            updates[self.mu] = T.clip(updates[self.mu],self.min_mu,self.max_mu)

        if self.B_driver in updates:
            updates[self.B_driver] = T.clip(updates[self.B_driver],self.min_B,self.max_B)

        if self.bias_hid in updates:
            updates[self.bias_hid] = T.clip(updates[self.bias_hid],self.min_bias_hid,self.max_bias_hid)

        if self.hard_max_step is not None:
            for param in updates:
                updates[param] = T.clip(updates[param],param-self.hard_max_step,param+self.hard_max_step)


    @classmethod
    def expected_log_prob_vhs_needed_stats(cls):
        h = S3C.log_likelihood_h_needed_stats()
        s = S3C.log_likelihood_s_given_h_needed_stats()
        v = S3C.expected_log_prob_v_given_hs_needed_stats()

        union = h.union(s).union(v)

        return union


    def expected_log_prob_vhs(self, stats):

        expected_log_prob_v_given_hs = self.expected_log_prob_v_given_hs(stats)
        #log_likelihood_v_given_hs = Print('log_likelihood_v_given_hs')(log_likelihood_v_given_hs)
        log_likelihood_s_given_h  = self.log_likelihood_s_given_h(stats)
        #log_likelihood_s_given_h = Print('log_likelihood_s_given_h')(log_likelihood_s_given_h)
        log_likelihood_h          = self.log_likelihood_h(stats)
        #log_likelihood_h = Print('log_likelihood_h')(log_likelihood_h)

        rval = expected_log_prob_v_given_hs + log_likelihood_s_given_h + log_likelihood_h

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
        second_hs = stats.d['second_hs']
        mean_D_sq_mean_Q_hs = stats.d['mean_D_sq_mean_Q_hs']

        term1 = 0.5 * T.sqr(NH) * T.sum(T.log(self.B))
        #term1 = Print('term1')(term1)
        term2 = 0.5 * (NH + 1) * T.dot(self.B,T.dot(self.W,mean_sq_hs))
        #term2 = Print('term2')(term2)
        term3 = - (self.B *  ( second_hs.dimshuffle('x',0,1) * self.W.dimshuffle(0,1,'x') *
                        self.W.dimshuffle(0,'x',1)).sum(axis=(1,2))).sum()
        #term3 = Print('term3')(term3)
        a = T.dot(T.sqr(self.W), mean_D_sq_mean_Q_hs)
        term4 = -0.5 * T.dot(self.B, a)
        #term4 = Print('term4')(term4)

        rval = term1 + term2 + term3 + term4

        return rval

    @classmethod
    def expected_log_prob_v_given_hs_needed_stats(cls):
        return set(['mean_sq_v','mean_hsv','second_hs'])

    def log_prob_v_given_hs(self, V, H, Mu1):
        """
        V, H, Mu1 are SAMPLES   (i.e., H must be LITERALLY BINARY)
        Return value is a vector, of length batch size
        """

        half = as_floatX(0.5)
        two = as_floatX(2.)
        pi = as_floatX(np.pi)
        N = as_floatX(self.nhid)

        term1 = half * T.sum(T.log(self.B))
        term2 = - half * N * T.log(two * pi)

        mean_HS = H * Mu1
        recons = T.dot(H*Mu1, self.W.T)
        residuals = V - recons

        term3 = - half * T.dot(T.sqr(residuals), self.B)

        rval = term1 + term2 + term3

        assert len(rval.type.broadcastable) == 1

        return rval


    def expected_log_prob_v_given_hs(self, stats):
        """
        Return value is a SCALAR-- expectation taken across batch index too
        """


        """
        E_v,h,s \sim Q log P( v | h, s)
        = E_v,h,s \sim Q log sqrt(B/2 pi) exp( - 0.5 B (v- W[v,:] (h*s) )^2)
        = E_v,h,s \sim Q 0.5 log B - 0.5 log 2 pi - 0.5 B v^2 + v B W[v,:] (h*s) - 0.5 B sum_i sum_j W[v,i] W[v,j] h_i s_i h_j s_j
        = 0.5 log B - 0.5 log 2 pi - 0.5 B v^2 + v B W[v,:] (h*s) - 0.5 B sum_i,j W[v,i] W[v,j] cov(h_i s_i, h_j s_j)
        """


        half = as_floatX(0.5)
        two = as_floatX(2.)
        pi = as_floatX(np.pi)
        N = as_floatX(self.nhid)

        mean_sq_v = stats.d['mean_sq_v']
        mean_hsv  = stats.d['mean_hsv']
        second_hs = stats.d['second_hs']

        term1 = half * T.sum(T.log(self.B))
        term2 = - half * N * T.log(two * pi)
        term3 = - half * T.dot(self.B, mean_sq_v)
        term4 = T.dot(self.B , (self.W * mean_hsv.T).sum(axis=1))
        term5 = - half * T.dot(self.B,  ( second_hs.dimshuffle('x',0,1) * self.W.dimshuffle(0,1,'x') *
                        self.W.dimshuffle(0,'x',1)).sum(axis=(1,2)))

        rval = term1 + term2 + term3 + term4 + term5

        assert len(rval.type.broadcastable) == 0

        return rval

    @classmethod
    def log_likelihood_s_given_h_needed_stats(cls):
        return set(['mean_h','mean_hs','mean_sq_s'])

    def log_likelihood_s_given_h(self, stats):

        warnings.warn('This function has NOT been verified by the finite sample method')

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

        warnings.warn('This function has not been verified by the finite sample method')

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


    def make_B_and_w(self):
        if self.tied_B:
            #can't just use a dimshuffle; dot products involving B won't work
            #and because doing it this way makes the partition function multiply by nvis automatically
            self.B = self.B_driver + as_floatX(np.zeros(self.nvis))
        else:
            self.B = self.B_driver

        self.w = T.dot(self.B, T.sqr(self.W))

    def redo_theano(self):
        init_names = dir(self)

        self.make_B_and_w()

        self.get_B_value = function([], self.B)


        X = T.matrix(name='V')
        X.tag.test_value = np.cast[config.floatX](self.rng.randn(self.test_batch_size,self.nvis))
        print 'made X test value with shape ',X.tag.test_value.shape

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

        if True:#self.monitor.examples_seen % 10000 == 0:

            print ""
            b = self.bias_hid.get_value(borrow=True)
            assert not np.any(np.isnan(b))
            p = 1./(1.+np.exp(-b))
            print 'p: ',(p.min(),p.mean(),p.max())
            B = self.B_driver.get_value(borrow=True)
            assert not np.any(np.isnan(B))
            print 'B: ',(B.min(),B.mean(),B.max())
            mu = self.mu.get_value(borrow=True)
            assert not np.any(np.isnan(mu))
            print 'mu: ',(mu.min(),mu.mean(),mu.max())
            alpha = self.alpha.get_value(borrow=True)
            assert not np.any(np.isnan(alpha))
            print 'alpha: ',(alpha.min(),alpha.mean(),alpha.max())
            W = self.W.get_value(borrow=True)
            assert not np.any(np.isnan(W))
            print 'W: ',(W.min(),W.mean(),W.max())
        if self.debug_m_step:
            if self.em_functional_diff.get_value() < 0.0:
                print "m step decreased the em functional"
                print self.em_functional_diff.get_value()
                quit(-1)
    #

    def get_weights_format(self):
        return ['v','h']

class E_step(object):
    def __init__(self):
        self.model = None

    def get_monitoring_channels(self, V, model):
        return {}

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

        The update equations were derived based on updating h_i and
        s_i simultaneously, but not updating all units simultaneously.

        The updates are not valid for updating h_i without also updating
        h_i (i.e., doing this could increase the KL divergence).

        They are also not valid for updating all units simultaneously,
        but we do this anyway.

        """

    def truncated_KL(self, V, model, obs):
        """ KL divergence between variation and true posterior, dropping terms that don't
            depend on the mean field parameters """

        H = obs['H']
        sigma0 = obs['sigma0']
        Sigma1 = obs['Sigma1']
        Mu1 = obs['Mu1']
        mu0 = obs['mu0']

        entropy_term = - model.entropy_hs(H = H, sigma0 = sigma0, Sigma1 = Sigma1)
        energy_term = model.expected_energy_vhs(V, H, mu0, Mu1, sigma0, Sigma1)

        KL = entropy_term + energy_term

        return KL

    def em_functional(self, V, model, obs):
        """ Return value is a scalar """

        needed_stats = S3C.expected_log_prob_vhs_needed_stats()

        stats = SufficientStatistics.from_observations( needed_stats = needed_stats,
                                                        X = V, ** obs )

        H = obs['H']
        sigma0 = obs['sigma0']
        Sigma1 = obs['Sigma1']

        entropy_term = (model.entropy_hs(H = H, sigma0 = sigma0, Sigma1 = Sigma1)).mean()
        likelihood_term = model.expected_log_prob_vhs(stats)

        em_functional = entropy_term + likelihood_term

        return em_functional


    def get_monitoring_channels(self, V, model):

        rval = {}

        if self.monitor_kl or self.monitor_em_functional:
            obs_history = self.mean_field(V, return_history = True)

            for i in xrange(1, 2 + len(self.h_new_coeff_schedule)):
                obs = obs_history[i-1]
                if self.monitor_kl:
                    rval['trunc_KL_'+str(i)] = self.truncated_KL(V, model, obs).mean()
                if self.monitor_em_functional:
                    rval['em_functional_'+str(i)] = self.em_functional(V, model, obs).mean()

        return rval


    def __init__(self, h_new_coeff_schedule, monitor_kl = False, monitor_em_functional = False):
        """Parameters
        --------------
        h_new_coeff_schedule:
            list of coefficients to put on the new value of h on each damped mean field step
                    (coefficients on s are driven by a special formula)
            length of this list determines the number of mean field steps
        """

        self.h_new_coeff_schedule = h_new_coeff_schedule
        self.monitor_kl = monitor_kl
        self.monitor_em_functional = monitor_em_functional

        super(VHS_E_Step, self).__init__()

    def init_mf_H(self, V):
        if self.model.recycle_q:
            rval = self.model.prev_H
        else:
            #just use the prior
            value =  T.nnet.sigmoid(self.model.bias_hid)
            rval = T.alloc(value, V.shape[0], value.shape[0])

        return rval

    def init_mf_Mu1(self, V):
        if self.model.recycle_q:
            rval = self.model.prev_Mu1
        else:
            #just use the prior
            value = self.model.mu
            if config.compute_test_value != 'off':
                assert value.tag.test_value != None
            rval = T.alloc(value, V.shape[0], value.shape[0])

        return rval



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

        V_name = make_name(V, 'anon_V')
        H_name = make_name(H, 'anon_H')
        Mu1_name = make_name(Mu1, 'anon_Mu1')

        A.name = 'mean_field_A( %s, %s, %s ) ' % ( V_name, H_name, Mu1_name)

        return A

    def mean_field_Mu1(self, A):

        alpha = self.model.alpha
        w = self.model.w

        denom = alpha + w

        Mu1 =  A / denom

        A_name = make_name(A, 'anon_A')

        Mu1.name = 'mean_field_Mu1(%s)'%A_name

        return Mu1

    def mean_field_Sigma1(self):
        """TODO: this is a bad name, since in the univariate case we would
         call this sigma^2
        I think what I was going for was covariance matrix Sigma constrained to be diagonal
         but it is still confusing """

        rval =  1./ (self.model.alpha + self.model.w )

        rval.name = 'mean_field_Sigma1'

        return rval

    def mean_field_H(self, A):

        half = as_floatX(.5)
        alpha = self.model.alpha
        w = self.model.w

        term1 = half * T.sqr(A) / (alpha + w)

        term2 = self.model.bias_hid

        term3 = - half * T.sqr(self.model.mu) * self.model.alpha

        term4 = -half * T.log(self.model.alpha + self.model.w)

        term5 = half * T.log(self.model.alpha)

        arg_to_sigmoid = term1 + term2 + term3 + term4 + term5

        H = T.nnet.sigmoid(arg_to_sigmoid)

        A_name = make_name(A, 'anon_A')

        H.name = 'mean_field_H('+A_name+')'

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

    def mean_field(self, V, return_history = False):
        """

            return_history: if True:
                                returns a list of dictionaries with
                                showing the history of the mean field
                                parameters
                                throughout fixed point updates
                            if False:
                                returns a dictionary containing the final
                                mean field parameters
        """

        alpha = self.model.alpha

        sigma0 = 1. / alpha
        Sigma1 = self.mean_field_Sigma1()
        mu0 = T.zeros_like(sigma0)



        H   =    self.init_mf_H(V)
        Mu1 =    self.init_mf_Mu1(V)

        def make_dict():

            return {
                    'H' : H,
                    'mu0' : mu0,
                    'Mu1' : Mu1,
                    'sigma0' : sigma0,
                    'Sigma1': Sigma1,
                    }

        history = [ make_dict() ]

        for new_coeff in self.h_new_coeff_schedule:

            A = self.mean_field_A(V = V, H = H, Mu1 = Mu1)
            new_Mu1 = self.mean_field_Mu1(A = A)
            new_H = self.mean_field_H(A = A)

            H = self.damp_H(H = H, new_H = new_H, new_coeff = new_coeff)
            Mu1 = self.damp_Mu1(Mu1 = Mu1, new_Mu1 = new_Mu1)

            history.append(make_dict())

        if return_history:
            return history
        else:
            return history[-1]




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
        assert Mu1.tag.test_value != None

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

        stats = SufficientStatistics.from_observations(needed_stats = S3C.expected_log_prob_vhs_needed_stats(),
                X = V, **hid_observations)

        obj = model.expected_log_prob_vhs(stats)

        return { 'expected_log_prob_vhs' : obj }

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
            model.B_driver: step(model.B_driver, B)
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

    def needed_stats(self):
        return S3C.solve_vhsu_needed_stats()

    def get_updates(self, model, stats):

        W, bias_hid, alpha, mu, B = model.solve_vhsu_from_stats(stats)

        learning_updates = take_step(model, W, bias_hid, alpha, mu, B, self.new_coeff)

        return learning_updates

class VHS_Grad_M_Step(VHS_M_Step):

    def __init__(self, learning_rate):
        self.learning_rate = np.cast[config.floatX](float(learning_rate))

    def get_updates(self, model, stats):

        params = model.get_params()

        obj = model.expected_log_prob_vhs(stats)

        grads = T.grad(obj, params, consider_constant = stats.d.values())

        updates = {}

        for param, grad in zip(params, grads):
            updates[param] = param + self.learning_rate * grad

        return updates

    def needed_stats(self):
        return S3C.expected_log_prob_vhs_needed_stats()

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



