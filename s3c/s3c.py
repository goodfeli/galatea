from pylearn2.models.model import Model
from theano import config, function, shared
import theano.tensor as T
import numpy as N
floatX = config.floatX
from theano_linalg.ops import alloc_diag, extract_diag, matrix_inverse, pseudo_inverse
from theano.printing import Print
from scipy.linalg import inv
config.compute_test_value = 'raise'

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

        H = T.nnet.sigmoid(self.b + 0.5 * sq_term / beta  - 0.5 * log_term )

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


    def learn_params(self, X, H, mu0, Mu1, sigma0, Sigma1):


        #Solve multiple linear regression problem where
        # W is a matrix used to predict v from h*s

        mean_HS = H * Mu1


        outer = T.dot(mean_HS.T,mean_HS)


        diag = T.sum(H*(Sigma1+T.sqr(Mu1)),axis=0)

        mask = T.identity_like(outer)
        masked_outer = (1-mask)*outer

        #extracted_diag = extract_diag(outer)

        eps = 1e-6
        #floored_diag = T.clip(extracted_diag, eps, 1e30)
        #xtx = masked_outer + floored_diag
        xtx = (1-mask)*outer + alloc_diag(diag+eps)

        xtx_inv =  matrix_inverse(xtx)

        #print "WARNING: still not really the right thing, b/c of issue with diag."

        W = T.dot(xtx_inv,T.dot(mean_HS.T,X)).T

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

        #debugging hack
        self.mse_shared = sharedX(0)
        learning_updates[self.mse_shared] = self.mse
        self.tsq_shared = sharedX(0)
        learning_updates[self.tsq_shared] = self.tsq
        """self.H_shared = sharedX(N.zeros((1,1)))
        self.Wres_shared = sharedX(N.zeros((1,1)))
        self.X_shared = sharedX(N.zeros((1,1)))
        learning_updates[self.H_shared] = self.H
        learning_updates[self.Wres_shared] = self.Wres
        learning_updates[self.X_shared] = self.X
        """

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
        init_names = dir(self)

        self.w = T.dot(self.B, T.sqr(self.W))

        X = T.matrix()
        X.tag.test_value = N.cast[floatX](self.rng.randn(5,self.nvis))

        self.learn_func = self.make_learn_func(X)

        final_names = dir(self)

        self.register_names_to_del([name for name in final_names if name not in init_names])
    #

    def learn(self, dataset, batch_size):
        self.learn_mini_batch(dataset.get_batch_design(batch_size))
    #

    def learn_mini_batch(self, X):
        #print "batch mean removal hack"
        #X -= X.mean(axis=0)

        print 'mean mag of mean feature value '
        print N.abs(X.mean(axis=0)).mean()


        self.learn_func(X)

        #debugging hack
        print 'mse: ',self.mse_shared.get_value()
        print 'tsq: ',self.tsq_shared.get_value()

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

        #assert False
    #

    def get_weights_format(self):
        return ['v','h']
    #
#

