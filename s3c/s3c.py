from pylearn2.models.model import Model
from theano import config, function, shared
import theano.tensor as T
import numpy as N
floatX = config.floatX
from theano.sandbox.linalg.ops import alloc_diag, extract_diag, matrix_inverse
from theano.printing import Print
from scipy.linalg import inv
#config.compute_test_value = 'raise'

def sharedX(X, name):
    return shared(N.cast[floatX](X),name=name)

class SufficientStatisticsHolder:
    def __init__(self, nvis, nhid):
        self.d = {
                    "mean_h" :      sharedX(N.zeros(nhid), "mean_h" ),
                    "mean_v" :      sharedX(N.zeros(nvis), "mean_v" ),
                    "mean_sq_v" :   sharedX(N.zeros(nvis), "mean_sq_v" ),
                    "mean_s1"   :   sharedX(N.zeros(nhid), "mean_s1"),
                    "mean_s"    :   sharedX(N.zeros(nhid), "mean_s" ),
                    "mean_sq_s" :   sharedX(N.zeros(nhid), "mean_sq_s" ),
                    "mean_hs" :     sharedX(N.zeros(nhid), "mean_hs" ),
                    "mean_sq_hs" :  sharedX(N.zeros(nhid), "mean_sq_hs" ),
                    "cov_hs" :      sharedX(N.zeros((nhid,nhid)), 'cov_hs'),
                    "mean_hsv" :    sharedX(N.zeros((nhid,nvis)), 'mean_hsv')
                }

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
    def from_observations(self, X, H, mu0, Mu1, sigma0, Sigma1):

        m = T.cast(X.shape[0],floatX)

        #mean_h
        assert H.dtype == floatX
        mean_h = T.mean(H, axis=0)
        assert H.dtype == mean_h.dtype
        assert mean_h.dtype == floatX

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

        #mean_sq_hs
        mean_sq_HS = H * (Sigma1 + T.sqr(Mu1))
        mean_sq_hs = T.mean(mean_sq_HS, axis=0)

        #cov_hs
        outer = T.dot(mean_HS.T,mean_HS) /m
        mask = T.identity_like(outer)
        cov_hs = (1.-mask) * outer + alloc_diag(mean_sq_hs)

        #mean_hsv
        mean_hsv = T.dot(mean_HS.T,X) / m



        d = {
                    "mean_h"        :   mean_h,
                    "mean_v"        :   mean_v,
                    "mean_sq_v"     :   mean_sq_v,
                    "mean_s"        :   mean_s,
                    "mean_s1"       :   mean_s1,
                    "mean_sq_s"     :   mean_sq_s,
                    "mean_hs"       :   mean_hs,
                    "mean_sq_hs"    :   mean_sq_hs,
                    "cov_hs"        :   cov_hs,
                    "mean_hsv"      :   mean_hsv
                }

        for key in d:
            d[key].name = 'observed_'+key

        return SufficientStatistics(d)

    def decay(self, coeff):
        rval_d = {}

        coeff = N.cast[floatX](coeff)

        for key in self.d:
            rval_d[key] = self.d[key] * coeff
            rval_d[key].name = 'decayed_'+self.d[key].name
        #

        return SufficientStatistics(rval_d)

    def accum(self, coeff, stats):

        if hasattr(coeff,'dtype'):
            assert coeff.dtype == floatX
        else:
            assert isinstance(coeff,float)
            coeff = N.cast[floatX](coeff)

        rval_d = {}

        for key in self.d:
            rval_d[key] = self.d[key] + coeff * stats.d[key]
            rval_d[key].name = 'blend_'+self.d[key].name+'_'+stats.d[key].name

        return SufficientStatistics(rval_d)




class S3C(Model):
    def __init__(self, nvis, nhid, irange, init_bias_hid,
                       init_B, min_B, max_B,
                       init_alpha, min_alpha, max_alpha, init_mu, N_schedule,
                       step_space,
                       step_scale, W_eps = 1e-6, mu_eps = 1e-8,
                        min_bias_hid = -1e30, max_bias_hid = 1e30,
                       learn_after = None):
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
        N_schedule: list of values to use for N throughout mean field updates.
                    len(N_schedule) determines # mean field steps
        step_scale: Exponential decay steps on a variable eta take the form
                        eta:=  step_scale * new_observation + (1-step_scale) * eta
        step_space: Possible values:
                    "suff_stat" : exponentially decay sufficient statistics,
                                  jump to new parameter solution on each batch
                    "params" : recompute sufficient statistics on each batch,
                                 exponentially decay parameters
        W_eps:       L2 regularization parameter for linear regression problem for W
        mu_eps:      L2 regularization parameter for linear regression problem for b
        learn_after: only applicable for step_space = 'suff_stat'
                        begins learning parameters and decaying sufficient statistics
                        after seeing learn_after examples
                        until this time, only accumulates sufficient statistics
        """

        super(S3C,self).__init__()

        self.W_eps = N.cast[floatX](float(W_eps))
        self.mu_eps = N.cast[floatX](float(mu_eps))
        self.nvis = nvis
        self.nhid = nhid
        self.irange = irange
        self.init_bias_hid = init_bias_hid
        self.init_alpha = init_alpha
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.init_B = init_B
        self.min_B = min_B
        self.max_B = max_B
        self.N_schedule = N_schedule
        self.step_scale = N.cast[floatX](float(step_scale))
        self.init_mu = init_mu
        self.min_bias_hid = min_bias_hid
        self.max_bias_hid = max_bias_hid

        assert step_space in ['suff_stat', 'params']
        if step_space == 'suff_stat':
            assert learn_after is not None
            self.stat_space_step = True
        else:
            self.stat_space_step = False
            assert learn_after is None
        #

        self.step_space = step_space
        self.learn_after = learn_after

        self.reset_rng()

        self.redo_everything()

    def reset_rng(self):
        self.rng = N.random.RandomState([1.,2.,3.])

    def redo_everything(self):

        self.W = sharedX(self.rng.uniform(-self.irange, self.irange, (self.nvis, self.nhid)), name = 'W')
        self.bias_hid = sharedX(N.zeros(self.nhid)+self.init_bias_hid, name='bias_hid')
        self.alpha = sharedX(N.zeros(self.nhid)+self.init_alpha, name = 'alpha')
        self.mu = sharedX(N.zeros(self.nhid)+self.init_mu, name='mu')
        self.B = sharedX(N.zeros(self.nvis)+self.init_B, name='B')

        self.suff_stat_holder = SufficientStatisticsHolder(nvis = self.nvis, nhid = self.nhid)

        self.redo_theano()
    #


    def learn_from_stats(self, stats):
        assert self.stat_space_step

        #Solve multiple linear regression problem where
        # W is a matrix used to predict v from h*s


        #cov_hs[i,j] = E_D,Q h_i s_i h_j s_j   (note that diagonal has different formula)
        cov_hs = stats.d['cov_hs']
        #mean_hsv[i,j] = E_D,Q h_i s_i v_j
        mean_hsv = stats.d['mean_hsv']

        regularized = cov_hs + alloc_diag(T.ones_like(self.mu) * self.W_eps)

        inv = matrix_inverse(regularized)

        W = T.dot(inv,mean_hsv).T

        # B is the precision of the residuals
        # variance of residuals:
        # var( [W hs - t]_i ) =
        # var( W_i hs ) + var( t_i ) + 2 ( mean( W_i hs ) mean(t_i) - mean( W_i hs t_i ) )
        # = var_recons + var_target + 2 ( mean_recons * mean_target - mean_recons_target )


        #mean_v[i] = E_D[ v_i ]
        mean_v = stats.d['mean_v']
        #mean_sq_v[i] = E_D[ v_i ^2 ]
        mean_sq_v = stats.d['mean_sq_v']
        var_v = mean_sq_v - T.sqr(mean_v)

        #mean_hs[i] = E_D,Q[ h_i s_i ]
        mean_hs = stats.d['mean_hs']
        mean_recons = T.dot(W, mean_hs)

        #mean_sq_recons[i] = E_D,Q [ ( W h \circ s)[i]^2 ]
        #E_D,Q  [   (W h \circ s )_i ^2 ]
        #= E_D,Q  [   (W_i: h \circ s )^2  ]
        #= E_D,Q  [   \sum_j \sum_k W_ij h_j s_j W_ik h_k s_k    ]
        #= E_D,Q  [   \sum_j W_ij^2 h_j s_j^2    \sum_{k \neq j} W_ij h_j s_j W_ik h_k s_k    ]
        #=    \sum_j W_ij^2  E_D,Q  [ h_j s_j^2 ]   \sum_{k \neq j} E_D,Q  [ W_ij h_j s_j W_ik h_k s_k    ]
        #=    \sum_j W_ij^2  mean_sq_hs[j] +   \sum_{k \neq j} E_D,Q  [ W_ij h_j s_j W_ik h_k s_k    ]
        #=    T.dot(T.sqr(W),  mean_sq_hs)[i] +  \sum_j  \sum_{k \neq j} E_D,Q  [ W_ij h_j s_j W_ik h_k s_k    ]
        #=    T.dot(T.sqr(W),  mean_sq_hs)[i] +  \sum_j  \sum_{k \neq j} W_ij  W_ik cov_hs[j,k]    ]
        #(cov_hs is E_D E_Q hs, ie hs[j] and hs[k] are not independent in this distribution, since the way the distributional particles line up for different examples can correlate them)
        #= (cov_hs.dimshuffle('x',0,1) * W.dimshuffle(0,'x',1) * W.dimshuffle(0,1,'x')).sum(axis=(1,2))
        #TODO: is there a more efficient way to do this?
        W_out = W.dimshuffle(0,'x',1) * W.dimshuffle(0,1,'x')
        mean_sq_recons = (cov_hs.dimshuffle('x',0,1) * W_out).sum(axis=(1,2))

        var_recons = mean_sq_recons - T.sqr(mean_recons)

        #mean_recons_v[i] = E_D,Q [ v[i] W[i,:] h \circ s ]
        #                 = E_D,Q [ v[i] \sum_j W_ij h_j s_j ]
        #                 = \sum_j W_ij E_D,Q[ v[i] h[j] s[j] ]
        #
        mean_recons_v = (W * mean_hsv.T).sum(axis=1)



        var_residuals = var_recons + var_v + N.cast[floatX](2.) * ( mean_recons * mean_v - mean_recons_v)


        B = 1. / var_residuals


        # Now a linear regression problem where mu_i is used to predict
        # s_i from h_i

        # mu_i = ( h^T h + reg)^-1 h^T s_i

        mean_h = stats.d['mean_h']
        assert mean_h.dtype == floatX
        reg = self.mu_eps
        mu = mean_hs/(mean_h+reg)

        #todo: get rid of mean_s1

        var_mu_h = T.sqr(mu) * (mean_h*(1.-mean_h))
        mean_sq_s = stats.d['mean_sq_s']
        mean_s = stats.d['mean_s']
        var_s = mean_sq_s - T.sqr(mean_s)


        mean_mu_h = mu * mean_h
        mean_mu_h_s = mu * mean_hs

        var_s_resid = var_mu_h + var_s + 2. * (mean_mu_h * mean_s - mean_mu_h_s)

        alpha = 1. / var_s_resid


        #probability of hiddens just comes from sample counting
        #to put it back in bias_hid space apply sigmoid inverse

        p = T.clip(mean_h,N.cast[floatX](1e-8),N.cast[floatX](1.-1e-8))

        assert p.dtype == floatX

        bias_hid = T.log( - p / (p-1.) )

        assert bias_hid.dtype == floatX

        return W, bias_hid, alpha, mu, B
    #



    def init_mf_H(self,V):
        assert type(self.N_schedule[-1]) == type(3.)
        assert self.N_schedule[-1] >= 1
        arg_to_log = 1.+(1./self.alpha) * self.N_schedule[-1] * self.w

        kklhlh = 0.5 * T.sqr (self.alpha*self.mu+T.dot(V*self.B,self.W)) / (self.alpha + self.w)

        H = T.nnet.sigmoid( kklhlh + self.bias_hid - 0.5 * T.log(arg_to_log) )
        return H
    #

    def init_mf_Mu1(self, V):
        Mu1 = (self.alpha*self.mu + T.dot(V*self.B,self.W))/(self.alpha+self.w)

        return Mu1
    #

    def mean_field_U(self, H, Mu1, N):
        prod = Mu1 * H

        first_term = T.dot(prod, self.W.T)
        first_term_broadcast = first_term.dimshuffle(0,'x',1)

        W_broadcast = self.W.dimshuffle('x',1,0)
        prod_broadcast = prod.dimshuffle(0,1,'x')

        second_term = N * W_broadcast * prod_broadcast

        U = first_term_broadcast - second_term

        return U
    #

    def mean_field_H(self, U, V, N):

        BW = self.W * (self.B.dimshuffle(0,'x'))

        filt = T.dot(V,BW)

        u_contrib = (U * BW.dimshuffle('x',1,0)).sum(axis=2)

        pre_sq = filt - u_contrib + self.alpha * self.mu

        sq_term = T.sqr(pre_sq)

        beta = self.alpha + N * self.w

        log_term = T.log(1.0 + N * self.w / self.alpha )

        H = T.nnet.sigmoid(self.bias_hid + 0.5 * sq_term / beta  - 0.5 * log_term )

        return H
    #

    def mean_field_Mu1(self, U, V, N):

        beta = self.alpha + N * self.w

        BW = self.W * self.B.dimshuffle(0,'x')

        filt = T.dot(V,BW)

        u_mod = - (U * BW.dimshuffle('x',1,0)).sum(axis=2)

        Mu1 = (filt + u_mod + self.alpha * self.mu) / beta

        return Mu1
    #


    def mean_field_Sigma1(self, N):
        Sigma1 = 1./(self.alpha + N * self.w)
        return Sigma1
    #


    def mean_field(self, V):
        sigma0 = 1. / self.alpha
        mu0 = T.zeros_like(sigma0)

        H   =    self.init_mf_H(V)
        Mu1 =    self.init_mf_Mu1(V)


        for N in self.N_schedule:
            U   = self.mean_field_U  (H = H, Mu1 = Mu1, N = N)
            H   = self.mean_field_H  (U = U, V = V,     N = N)
            Mu1 = self.mean_field_Mu1(U = U, V = V,     N = N)


        Sigma1 = self.mean_field_Sigma1(N = self.N_schedule[-1])

        return H, mu0, Mu1, sigma0, Sigma1
    #


    def learn_from_batch_obs(self, X, H, mu0, Mu1, sigma0, Sigma1):
        #assert not self.stat_space_step

        m = T.cast(X.shape[0],floatX)

        assert X.dtype == floatX

        #Solve multiple linear regression problem where
        # W is a matrix used to predict v from h*s

        mean_HS = H * Mu1
        mean_sq_HS = H * (Sigma1+T.sqr(Mu1))
        mean_mean_sq_HS = mean_sq_HS.mean(axis=0)
        sum_mean_sq_HS = mean_mean_sq_HS * T.cast(H.shape[0],dtype=floatX)

        outer = T.dot(mean_HS.T,mean_HS)


        diag = sum_mean_sq_HS
        assert diag.dtype == floatX

        mask = T.identity_like(outer)
        masked_outer = (N.cast[floatX](1)-mask)*outer
        assert masked_outer.dtype == floatX

        #extracted_diag = extract_diag(outer)

        eps = self.W_eps
        #floored_diag = T.clip(extracted_diag, eps, 1e30)
        #xtx = masked_outer + floored_diag
        final_diag = diag + eps
        assert final_diag.dtype == floatX
        diag_mat = alloc_diag(final_diag)
        assert diag_mat.dtype == floatX
        xtx = masked_outer + diag_mat

        assert xtx.dtype == floatX
        xtx = xtx / m

        xtx_inv =  matrix_inverse(xtx)
        assert xtx_inv.dtype == floatX

        #print "WARNING: still not really the right thing, b/c of issue with diag."

        W = T.dot(xtx_inv,T.dot(mean_HS.T,X/m)).T
        assert W.dtype == floatX

        #W = T.dot(pseudo_inverse(mean_HS),X).T

        #debugging hacks
        self.H = mean_HS
        self.Wres = W
        self.X = X
        residuals = T.dot(mean_HS, self.W.T) - X
        self.mse = T.mean(T.sqr(residuals))
        self.tsq = T.mean(T.sqr(X))


        # B is the precision of the residuals
        # variance of residuals:
        # var( [W hs - t]_i ) =
        # var( W_i hs ) + var( t_i ) + 2 ( mean( W_i hs ) mean(t_i) - mean( W_i hs t_i ) )
        # = var_recons + var_target + 2 ( mean_recons * mean_target - mean_recons_target )


        mean_target = T.mean(X,axis=0)
        assert mean_target.dtype == floatX
        mean_sq_target = T.mean(T.sqr(X),axis=0)
        var_target = mean_sq_target - T.sqr(mean_target)
        assert var_target.dtype == floatX

        mean_mean_HS = T.mean(mean_HS, axis=0)
        assert mean_mean_HS.dtype == floatX
        mean_recons = T.dot(W, mean_mean_HS)
        assert mean_recons.dtype == floatX

        W_sq = T.sqr(W)
        term1 = T.dot(W_sq, mean_mean_sq_HS)
        term2 = T.mean(T.sqr(T.dot(mean_HS,W.T)),axis=0)
        term3 = T.dot(W_sq,T.mean(T.sqr(mean_HS),axis=0))
        mean_sq_recons = term1 + term2 - term3
        assert mean_sq_recons.dtype == floatX

        #memory hog:
        #mean_sq_recons = T.dot(T.sqr(W), diag) + (W.dimshuffle(0,1,'x')*W.dimshuffle(0,'x',1)*masked_outer.dimshuffle('x',0,1)).sum(axis=(1,2))

        var_recons = mean_sq_recons - T.sqr(mean_recons)
        assert var_recons.dtype == floatX

        mean_recons_target = T.mean(X * T.dot(mean_HS,W.T), axis = 0)
        assert mean_recons_target.dtype == floatX



        var_residuals = var_recons + var_target + N.cast[floatX](2.) * ( mean_recons * mean_target - mean_recons_target)

        assert var_residuals.dtype == floatX

        B = 1. / var_residuals


        # Now a linear regression problem where mu_i is used to predict
        # s_i from h_i

        # mu_i = ( h^T h + reg)^-1 h^T s_i

        q = T.mean(H,axis=0)
        reg = self.mu_eps
        mu = T.mean(mean_HS,axis=0)/(q+reg)
        #mu = T.mean(Mu1,axis=0)

        var_mu_h = T.sqr(mu) * (q*(1.-q))
        var_s = T.mean( H * (Sigma1+T.sqr(Mu1)) + (1-H)*(sigma0+T.sqr(sigma0)) , axis=0)


        mean_mu_h = mu * q
        mean_s = T.mean(H*Mu1+(1.-H)*mu0,axis=0)
        mean_mu_h_s = mu * T.mean(mean_HS,axis=0)

        var_s_resid = var_mu_h + var_s + 2. * (mean_mu_h * mean_s - mean_mu_h_s)

        alpha = 1. / var_s_resid


        #probability of hiddens just comes from sample counting
        #to put it back in bias_hid space apply sigmoid inverse
        p = T.mean(H,axis=0)

        p = T.clip(p,1e-8,1.-1e-8)

        bias_hid = T.log( - p / (p-1.) )

        return W, bias_hid, alpha, mu, B
    #

    def make_learn_func(self, X, learn = None):
        """
        X: a symbolic design matrix
        learn:
            must be None unless taking steps in sufficient statistics space
            False: accumulate sufficient statistics
            True: exponentially decay sufficient statistics, accumulate new ones, and learn new params
        """

        #E step
        H, mu0, Mu1, sigma0, Sigma1 = self.mean_field(X)


        m = T.cast(X.shape[0],dtype = floatX)

        if self.stat_space_step:
            ######## Exponential decay in sufficient statistic space
            assert learn is not None

            old_stats = SufficientStatistics.from_holder(self.suff_stat_holder)
            new_stats = SufficientStatistics.from_observations(X, H, mu0, Mu1, sigma0, Sigma1)

            if learn:
                updated_stats = old_stats.decay(1.0-self.step_scale)
                updated_stats = updated_stats.accum(coeff = self.step_scale, stats = new_stats)

                #M step
                W, bias_hid, alpha, mu, B = self.learn_from_stats( updated_stats )

                learning_updates = {
                  self.W: W,
                  self.bias_hid: bias_hid,
                  self.alpha: alpha,
                  self.mu: mu,
                  self.B : B
                }
            else:
                updated_stats = old_stats.accum(coeff = m / self.learn_after, stats = new_stats)

                learning_updates = {}
            #

            self.suff_stat_holder.update(learning_updates, updated_stats)

        else:
            #######  Exponential decay in parameter space
            assert learn is None

            #M step
            W, bias_hid, alpha, mu, B = self.learn_from_batch_obs(X, H, mu0, Mu1, sigma0, Sigma1)

            #parameter updates-- don't do a full M step since we're using minibatches
            def step(old, new):
                return self.step_scale * new + (1.-self.step_scale) * old

            learning_updates = {
                    self.W: step(self.W, W),
                    self.bias_hid: step(self.bias_hid,bias_hid),
                    self.alpha: step(self.alpha, alpha),
                    self.mu: step(self.mu, mu),
                    self.B: step(self.B, B)
                    }

        self.censor_updates(learning_updates)

        #debugging hack
        #self.mse_shared = sharedX(0,'mse')
        #learning_updates[self.mse_shared] = self.mse
        #self.tsq_shared = sharedX(0,'tsq')
        #learning_updates[self.tsq_shared] = self.tsq

        return function([X], updates = learning_updates)
    #

    def censor_updates(self, updates):
        if self.alpha in updates:
            updates[self.alpha] = T.clip(updates[self.alpha],self.min_alpha,self.max_alpha)
        #

        if self.B in updates:
            updates[self.B] = T.clip(updates[self.B],self.min_B,self.max_B)
        #

        if self.bias_hid in updates:
            updates[self.bias_hid] = T.clip(updates[self.bias_hid],self.min_bias_hid,self.max_bias_hid)
        #
    #



    def log_likelihood_vhs(self, stats):
        """Note: drops some constant terms"""

        log_likelihood_v_given_hs = self.log_likelihood_v_given_hs(self, stats)
        log_likelihood_s_given_h  = self.log_likelihood_s_given_h(self, stats)
        log_likelihood_h          = self.log_likelihood_h(self, stats)

        rval = log_likelihood_v_given_hs + log_likelihood_s_given_h + log_likelihood_h

        return rval

    def log_likelihood_v_given_hs(self, stats):
        """Note: drops some constant terms"""

        mean_sq_v = stats.d['mean_sq_v']
        mean_hsv  = stats.d['mean_hsv']
        cov_hs = stats.d['cov_hs']

        term1 = 0.5 * T.log(self.B).sum()
        term2 = - 0.5 * T.dot(self.B, mean_sq_v)
        term3 = B * T.dot(mean_hsv,self.W).sum()
        term4 = -0.5 * B *  ( cov_hs.dimshuffle('x',0,1) * self.W.dimshuffle(0,1,'x') *
                        self.W.dimshuffle(0,'x',1)).sum()

        rval = term1 + term2 + term3 + term4

        return rval

    def log_likelihood_s_given_h(self, stats):
        """Note: drops some constant terms"""

        mean_h = stats.d['mean_h']
        mean_sq_s = stats.d['mean_sq_s']

        term1 = 0.5 * T.log( self.alpha ).sum()
        term2 = - 0.5 * T.dot( self.alpha , mean_sq_s )
        term3 = T.dot(self.mu*self.alpha*mean_h,1.-0.5 * self.mu)

        rval = term1 + term2 + term3

        return rval

    def log_likelihood_h(self, stats):
        mean_h = stats.d['mean_h']

        term1 = - T.dot(mean_h - 1., T.nnet.softplus(self.bias_hid))
        term2 = - T.dot(mean_h, T.nnet.softplus(-self.bias_hid))

        rval = term1 + term2

        return rval


    def redo_theano(self):
        init_names = dir(self)

        self.w = T.dot(self.B, T.sqr(self.W))

        X = T.matrix()
        X.tag.test_value = N.cast[floatX](self.rng.randn(5,self.nvis))

        if self.stat_space_step:
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


    def run_test(self, X):

        V = T.matrix()

        H, mu0, Mu1, sigma0, Sigma1 = self.mean_field(X)

        stats = SufficientStatistics.from_observations( V, H, mu0, Mu1, sigma0, Sigma1)

        batch_func = function([V], self.learn_from_batch_obs( V, H, mu0, Mu1, sigma0, Sigma1))
        stats_func = function([V], self.learn_from_stats(stats))

        batch_results = batch_func(X)
        stats_results = stats_func(X)


        assert len(batch_results) == len(stats_results)


        temp = batch_results[2]
        batch_results[2] = batch_results[3]
        batch_results[3] = batch_results[2]

        temp = stats_results[2]
        stats_results[2] = stats_results[3]
        stats_results[3] = stats_results[2]


        for i in xrange(len(batch_results)):
            if not N.allclose(batch_results[i],stats_results[i]):
                if i == 0 and N.abs(batch_results[i] - stats_results[i]).max() < 1e-4:
                    continue

                print 'parameter ',i,'does not match'
                print 'max diff: ',N.abs(batch_results[i]-stats_results[i]).max()
                assert False

        print "test passed"





    def learn_mini_batch(self, X):


        #self.run_test(X)


        if self.stat_space_step:
            if self.monitor.examples_seen >= self.learn_after:
                self.learn_func(X)
            else:
                self.accum_func(X)
        else:
            self.learn_func(X)
        #

        #debugging hack
        #print 'mse: ',self.mse_shared.get_value()
        #print 'tsq: ',self.tsq_shared.get_value()



        """W, H, X = self.Wres_shared.get_value(borrow=True), self.H_shared.get_value(borrow=True), self.X_shared.get_value(borrow=True)

        print 'diff W'
        print N.abs(W-
                    N.dot(inv(H),
                        N.dot(H.T,X)
                        ).T
                   ).max()
        print 'diff from ident'
        print N.abs(N.identity(H.shape[0])-N.dot(inv(N.dot(H.T,H)),N.dot(H.T,H))).max()

        print 'resid'
        print N.abs(X - N.dot(H, W.T) ).max()
        print 'proj resid'
        print N.abs(N.dot(H.T,N.dot(H,W.T))-N.dot(H.T,X)).max()
        """

        """cov_hs = self.suff_stat_holder.d['cov_hs'].get_value(borrow=True)
        a,b = N.linalg.eigh(cov_hs)

        assert not N.any(N.isnan(a))
        assert not N.any(N.isinf(a))
        print 'minimum eigenvalue: '+str(a.min())
        assert a.min() >= 0"""

        if self.monitor.examples_seen % 1000 == 0:
            B = self.B.get_value(borrow=True)
            print 'B: ',(B.min(),B.mean(),B.max())
            mu = self.mu.get_value(borrow=True)
            print 'mu: ',(mu.min(),mu.mean(),mu.max())
            alpha = self.alpha.get_value(borrow=True)
            print 'alpha: ',(alpha.min(),alpha.mean(),alpha.max())

    #

    def get_weights_format(self):
        return ['v','h']
    #
#

