from pylearn2.models.model import Model
from theano import config, function, shared
import theano.tensor as T
import numpy as N
floatX = config.floatX
from theano_linalg.ops import alloc_diag, matrix_inverse
from theano.printing import Print
#config.compute_test_value = 'raise'

def sharedX(X):
    return shared(N.cast[floatX](X))

class S3C(Model):
    def __init__(self, nvis, nhid, irange, init_bias_hid,
                       init_B,
                       init_alpha, init_mu, N_schedule,
                       step_scale ):
        super(S3C,self).__init__()

        self.nvis = nvis
        self.nhid = nhid
        self.irange = irange
        self.init_bias_hid = init_bias_hid
        self.init_alpha = init_alpha
        self.init_B = init_B
        self.N_schedule = N_schedule
        self.step_scale = step_scale
        self.init_mu = init_mu

        self.reset_rng()

        self.redo_everything()

    def reset_rng(self):
        self.rng = N.random.RandomState([1.,2.,3.])

    def redo_everything(self):

        self.W = sharedX(self.rng.uniform(-self.irange, self.irange, (self.nvis, self.nhid)))
        self.bias_hid = sharedX(N.zeros(self.nhid)+self.init_bias_hid)
        self.alpha = sharedX(N.zeros(self.nhid)+self.init_alpha)
        self.mu = sharedX(N.zeros(self.nhid)+self.init_mu)
        self.B = sharedX(N.zeros(self.nvis)+self.init_B)

        self.redo_theano()
    #

    def init_mf_H(self):
        assert type(self.N_schedule[-1]) == type(3.)
        assert self.N_schedule[-1] >= 1
        #self.w = Print('w')(self.w)
        arg_to_log = 1.+(1./self.alpha) * self.N_schedule[-1] * self.w
        #arg_to_log = Print('arg_to_log')(arg_to_log)
        h = T.nnet.sigmoid(self.bias_hid - 0.5 * T.log(arg_to_log) )
        H = h.dimshuffle('x',0)
        return H
    #

    def init_mf_Mu1(self):
        mu1 = self.mu
        Mu1 = mu1.dimshuffle('x',0)
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
        weighty_factor = (self.B.dimshuffle(0,'x')*self.W).dimshuffle('x',1,0)
        diff = V.dimshuffle(0,'x',1) - U

        tensordot = (weighty_factor * diff).sum(axis=2)

        C = T.sqr(tensordot)

        A = (0.5 / (self.alpha + N * self.w)) * C

        Z = A + self.bias_hid - 0.5 * T.log(1.+N * self.w/self.alpha)

        H = T.nnet.sigmoid(Z)

        return H
    #

    def mean_field_Mu1(self, U, V, H, N):

        BW = self.B.dimshuffle(0,'x')*self.W
        mlas = BW.dimshuffle('x',1,0) * H.dimshuffle(0,1,'x')
        diff = V.dimshuffle(0,'x',1) - U
        ioho = (mlas*diff).sum(axis=2)

        Mu1 = ioho / (self.alpha + N * self.w)

        return Mu1
    #


    def mean_field_Sigma1(self, H, N):
        Sigma1 = 1./(self.alpha + N * self.w*H)
        return Sigma1
    #


    def mean_field(self, V):
        mu0 = self.mu
        sigma0 = 1. / self.alpha

        H   =    self.init_mf_H()
        Mu1 =    self.init_mf_Mu1()

        #H = Print('H')(H)

        for N in self.N_schedule:
            U   = self.mean_field_U  (H = H, Mu1 = Mu1, N = N)
            H   = self.mean_field_H  (U = U, V = V,     N = N)
            #H = Print('H')(H)
            Mu1 = self.mean_field_Mu1(U = U, V = V, H = H,     N = N)

        Sigma1 = self.mean_field_Sigma1(H = H, N = self.N_schedule[-1])

        return H, mu0, Mu1, sigma0, Sigma1
    #


    def learn_params(self, X, H, mu0, Mu1, sigma0, Sigma1):
        #Solve multiple linear regression problem where
        # W is a matrix used to predict v from h*s

        mean_HS = H * Mu1
        outer = T.dot(mean_HS.T,mean_HS)


        diag = T.sum(H*(Sigma1+T.sqr(Mu1)),axis=0)

        mask = T.identity_like(outer)
        masked_outer = (1-mask)*outer
        eps = 1e-3
        xtx = (1-mask)*outer + alloc_diag(diag+eps)

        #xtx = Print('xtx')(xtx)

        xtx_inv = matrix_inverse(xtx)

        W = T.dot(xtx_inv,T.dot(mean_HS.T,X)).T


        # B is the precision of the residuals
        # variance of residuals:
        # var( [W hs - t]_i ) =
        # var( W_i hs ) + var( t_i ) + 2 ( mean( W_i hs ) mean(t_i) - mean( W_i hs t_i ) )
        # = var_recons + var_target + 2 ( mean_recons * mean_target - mean_recons_target )


        mean_target = T.mean(X,axis=0)
        mean_sq_target = T.mean(T.sqr(X),axis=0)
        var_target = mean_sq_target - T.sqr(mean_target)

        mean_hid = T.mean(mean_HS, axis=0)
        mean_recons = T.dot(self.W, mean_hid)
        mean_sq_recons = T.dot(T.sqr(self.W), diag) + (W.dimshuffle(0,1,'x')*W.dimshuffle(0,'x',1)*masked_outer.dimshuffle('x',0,1)).sum(axis=(1,2))

        var_recons = mean_sq_recons - T.sqr(mean_recons)

        mean_recons_target = T.mean(X * T.dot(mean_HS,self.W.T), axis = 0)

        var_residuals = var_recons + var_target + 2. * ( mean_recons * mean_target - mean_recons_target)

        B = 1. / var_residuals


        # Now a linear regression problem where mu_i is used to predict
        # s_i from h_i

        mu = T.mean(Mu1)
        q = T.mean(H,axis=0)

        var_mu_h = T.sqr(mu) * (q*(1.-q))
        var_s = T.mean( H * (Sigma1+T.sqr(Mu1)) + (1-H)*(sigma0+T.sqr(sigma0)) , axis=0)


        mean_mu_h = mu * q
        mean_s = T.mean(H*Mu1+(1.-H)*mu0)
        mean_mu_h_s = mu * T.mean(mean_HS,axis=0)

        var_s_resid = var_mu_h + var_s + 2. * (mean_mu_h * mean_s - mean_mu_h_s)

        alpha = 1. / var_s_resid


        #probability of hiddens just comes from sample counting
        #to put it back in bias_hid space apply sigmoid inverse
        p = T.mean(H,axis=0)

        p = T.clip(p,1e-3,1.-1e-3)

        bias_hid = T.log( - p / (p-1.) )

        return W, bias_hid, alpha, mu, B
    #

    def make_learn_func(self, X):

        #E step
        H, mu0, Mu1, sigma0, Sigma1 = self.mean_field(X)

        #M step
        W, bias_hid, alpha, mu, B = self.learn_params(X, H, mu0, Mu1, sigma0, Sigma1)

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

        return function([X], updates = learning_updates)
    #

    def censor_updates(self, updates):
        if self.alpha in updates:
            updates[self.alpha] = T.clip(updates[self.alpha],1e-3,1e5)
        #

        if self.B in updates:
            updates[self.B] = T.clip(updates[self.B],1e-3,1e5)
        #
    #



    def redo_theano(self):

        self.w = T.dot(self.B, T.sqr(self.W))

        X = T.matrix()
        X.tag.test_value = N.cast[floatX](self.rng.randn(5,self.nvis))

        self.learn_func = self.make_learn_func(X)
    #

    def learn(self, dataset, batch_size):
        self.learn_mini_batch(dataset.get_batch_design(batch_size))
    #

    def learn_mini_batch(self, X):
        self.learn_func(X)
    #
#

