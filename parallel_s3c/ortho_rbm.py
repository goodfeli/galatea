from pylearn2.models import Model
import numpy as np
from numpy.linalg import svd
from pylearn2.utils import sharedX
from pylearn2.costs.cost import Cost
import theano.tensor as T
from pylearn2.space import VectorSpace
import warnings
from theano.printing import Print
from pylearn2.train_extensions import TrainExtension
from theano.sandbox.rng_mrg import MRG_RandomStreams
from pylearn2.utils import block_gradient

def beta_orthogonalize(W, beta):

    sqrt_beta = np.sqrt(beta)

    W = (W.T * sqrt_beta).T
    print 'Computing SVD...'
    u, s, v = svd(W, full_matrices=False)
    print '...done'

    #print 'range of singular values: ',(s.min(), s.mean(), s.max())

    """
    uTu = np.dot(u.T,u)
    s = uTu.shape[0]
    if not np.allclose(uTu, np.identity(s)):
        print 'max err: ',np.abs(uTu-np.identity(s)).max()
        diagonal = np.zeros((s,))
        print 'uTu is not identity'
        for i in xrange(s):
            diagonal[i] = uTu[i,i]
            uTu[i,i] = 0
        print 'diagonal: ',(diagonal.min(), diagonal.mean(), diagonal.max())
        print 'off diagonal: ',(uTu.min(), uTu.mean(), uTu.max())
        assert False
    assert np.allclose(np.dot(v.T, v), np.identity(v.shape[1]))
    """

    W = np.dot(u, v)


    """
    print 'Checking that weights are orthogonal...'
    assert np.allclose(np.dot(W.T,W), np.identity(W.shape[1]))
    print '...done'
    """

    W = (W.T / sqrt_beta).T

    WBW = np.dot(W.T*beta, W)
    print 'Max error in W.T B W:', np.abs(WBW - np.identity(W.shape[1])).max()
    print 'Sum of squared errors:',np.square(WBW-np.identity(W.shape[1])).sum()

    return W

def random_ortho_columns(rows, columns, beta, rng=None):
    """
    Makes a random matrix where each column is unit norm
    and the columns are orthogonal to each other.
    """

    assert rows >= columns

    if rng is None:
        rng = np.random.RandomState([2012,11,13,3])

    W = rng.randn(rows, columns)

    W = beta_orthogonalize(W, beta)


    return W

class OrthoRBM(Model):

    def __init__(self, nvis, nhid, init_bias_hid, init_beta, init_scale, min_beta, fixed_point_orthogonalize = False, censor_beta_norms=True):

        self.__dict__.update(locals())
        del self.self

        self.rng = np.random.RandomState([2012,11,13])

        self.scale = sharedX(np.zeros((nhid,))+init_scale)
        self.scale.name = 'scale'

        self.b = sharedX(np.zeros((nhid,))+init_bias_hid)
        self.b.name = 'b'

        self.beta = sharedX(np.zeros(nvis,)+init_beta)
        self.beta.name = 'beta'

        self.W = sharedX(random_ortho_columns(nvis, nhid, self.beta.get_value(), self.rng))
        self.W.name = 'W'

        self._params = [self.scale, self.W, self.b, self.beta]
        self.input_space = VectorSpace(nvis)

    def log_likelihood(self, X):

        """

        log P(x) = log sum_h P(x,h)
                 = log sum_h exp(-E(x,h))/Z
                 = log sum_h exp(-E(x,h))/sum x' h' exp(-E(x',h'))
                 = log sum_h exp(-E(x,h)) - log sum x' h' exp(-E(x',h'))
                 = log sum_h exp(b^T h - 1/2 (x-Whs)^T beta (x-Whs)) - log sum x' h' exp(b^T h' - 1/2 (x'-Wh's)^T beta (x'-Wh's))
                 = log sum_h exp(b^T h - 1/2 x^T beta x -1/2 (Whs)^T beta (s*Wh) + x^T beta Whs )
                   - log sum x' sum_h' exp( " )
                = log sum_h exp(b^T h - 1/2 x^T beta x - 1/2 (sh)^T (sh) + x^T beta Whs)
                  - log sum x' sum_h' exp( " )
                = log exp(-1/2 x^T beta x) sum_h exp(b^T h - 1/2 (sh)^T sh + x^T beta Whs)
                  - log sum x' sum h' exp (b^T h' - 1/2 x'^T beta x' - 1/2 (sh')^T sh' + x'^T beta Wh's)
                = -1/2 x^T beta x + log sum_h exp(b^T h - 1/2 (sh)^T sh + x^T beta Whs )
                 - log sum x' exp(-1/2 x'^T beta x') sum h' Pi_i exp(b_i h'_i - 1/2 (s_i h'_i))^2+ x'^T beta W_:i h'_is_i)
                = -1/2 x^T beta x + log sum_h exp(b^T h - 1/2 (sh)^T sh + x^T beta Whs )
                 - log sum x' exp(-1/2 x'^T beta x') sum h' Pi_i exp(b_i h'_i - 1/2 (s_i h'_i))^2+ x'^T beta W_:i h'_is_i)
                = -1/2 x^T beta x + sum_i softplus(b_i - 1/2 s_i^2 + x^T beta W:i s_i)
                 - log sum x' exp(-1/2 x'^T beta x') Pi_i (1+exp(b_i - 1/2 s_i ^2+ x'^T beta W_:i s_i)
                = -1/2 x^T beta x + sum_i softplus(b_i - 1/2 s_i^2 + x^T beta W:i s_i)
                 - log sum x' exp(-1/2 x'^T beta x') sum h' Pi_i exp(b_i h'_i - 1/2 (s_i h'_i))^2+ x'^T beta W_:i h'_is_i)
                = -1/2 x^T beta x + sum_i softplus(b_i - 1/2 s_i^2 + x^T beta W:i s_i)
                  - log sum x' sum h' exp (b^T h' - 1/2 x'^T beta x' - 1/2 (sh')^T sh' + x'^T beta Wh's)
                = -1/2 x^T beta x + sum_i softplus(b_i - 1/2 s_i^2 + x^T beta W:i s_i)
                  - log sum h' exp(b^T h'-1/2 (sh')^T (sh') Pi_j sum_x' exp( - 1/2 beta x'_j^2  + x'_j beta_jj W_j:h's)
                = -1/2 x^T beta x + sum_i softplus(b_i - 1/2 s_i^2 + x^T beta W:i s_i)
                  - log sum h' exp(b^T h'-1/2 (sh')^T (sh') Pi_j sum_x' exp( - 1/2 beta_j (x'_j - W_j: h's)^2 + 1/2 beta_j (W_j:h's)^2)
                = -1/2 x^T beta x + sum_i softplus(b_i - 1/2 s_i^2 + x^T beta W:i s_i)
                  - log sum h' exp(b^T h'-1/2 (sh')^T (sh') Pi_j exp(1/2 beta_j (W_j: h's)^2 ) sum_x' exp( - 1/2 beta_j (x'_j - W_j: h's)^2)
                = -1/2 x^T beta x + sum_i softplus(b_i - 1/2 s_i^2 + x^T beta W:i s_i)
                  - log sum h' exp(b^T h'-1/2 (sh')^T (sh') Pi_j exp(1/2 beta_j (W_j: h's)^2 ) sqrt(beta_j / 2pi)

                = -1/2 x^T beta x + sum_i softplus(b_i - 1/2 s_i^2 + x^T beta W:i s_i)
                  +1/2 sum(log(beta)) - 1/2 sum(log(2*pi))
                  - log sum h' exp(b^T h'-1/2 (sh')^T (sh') Pi_j exp(1/2 beta_j (W_j: h's)^2 )
                = -1/2 x^T beta x + sum_i softplus(b_i - 1/2 s_i^2 + x^T beta W:i s_i)
                  +1/2 sum(log(beta)) - 1/2 sum(log(2*pi))
                  - log sum h' exp(b^T h'-1/2 (sh')^T (sh') + sum_j 1/2 beta_j (W_j: h's)^2 )
                = -1/2 x^T beta x + sum_i softplus(b_i - 1/2 s_i^2 + x^T beta W:i s_i)
                  +1/2 sum(log(beta)) - 1/2 sum(log(2*pi))
                  - log sum h' exp(b^T h'-1/2 (sh')^T (sh') + sum_j 1/2 sum_k beta_j sum_l W_jk W_jl  h'_k s_k h'_l s_l

                = -1/2 x^T beta x + sum_i softplus(b_i - 1/2 s_i^2 + x^T beta W:i s_i)
                  +1/2 sum(log(beta)) - 1/2 sum(log(2*pi))
                  - log sum h' exp(b^T h'-1/2 (sh')^T (sh') + 1/2 sum_j sum_k beta_j sum_l W_jk W_jl  h'_k s_k h'_l s_l

                = -1/2 x^T beta x + sum_i softplus(b_i - 1/2 s_i^2 + x^T beta W:i s_i)
                  +1/2 sum(log(beta)) - 1/2 sum(log(2*pi))
                  - log sum h' exp(b^T h'-1/2 (sh')^T (sh') + 1/2 h's^T W^T beta W hs )

                = -1/2 x^T beta x + sum_i softplus(b_i - 1/2 s_i^2 + x^T beta W:i s_i)
                  + 1/2 sum(log(beta)) - 1/2 sum(log(2*pi))
                  - log sum h' exp(b^T h'-1/2 (sh')^T (sh') + 1/2 h's^T hs )

                = -1/2 x^T beta x + sum_i softplus(b_i - 1/2 s_i^2 + x^T beta W:i s_i)
                  + 1/2 sum(log(beta)) - 1/2 sum(log(2*pi))
                  - log sum h' exp(b^T h' )
                = -1/2 x^T beta x + sum_i softplus(b_i - 1/2 s_i^2 + x^T beta W:i s_i)
                  +1/2 sum(log(beta)) - 1/2 sum(log(2*pi))
                  - log sum h' Pi_i exp(b_i h_i )
                = -1/2 x^T beta x + sum_i softplus(b_i - 1/2 s_i^2 + x^T beta W:i s_i)
                  +1/2 sum(log(beta)) - 1/2 sum(log(2*pi))
                  - log Pi_i 1 + exp(b_i )
                = -1/2 x^T beta x + sum_i softplus(b_i - 1/2 s_i^2 + x^T beta W:i s_i)
                  +1/2 sum(log(beta)) - 1/2 nvis log(2*pi)
                  - sum_i softplus(b_i)

        """

        term1 = -.5 * T.dot(T.sqr(X), self.beta)
        Z = self.b - 0.5 * T.sqr(self.scale) + (T.dot(X * self.beta, self.W) * self.scale)
        term2 = T.nnet.softplus(Z).sum(axis=1)
        beta = self.beta
        term3 = 0.5 * T.log(beta).sum()
        term4 = - 0.5 * T.log(2*np.pi) * self.nvis
        term5 = - T.nnet.softplus(self.b).sum()

        return term1 + term2 + term3 + term4 + term5

    def censor_updates(self, updates):
        if self.beta in updates:
            mask = self.beta > self.min_beta
            masked = self.beta * mask + (1-mask)*self.min_beta
            updates[self.beta] = masked
        if self.W in updates:
            W = updates[self.W]
            if self.beta in updates:
                beta = updates[self.beta]
            else:
                beta = self.beta

            if self.fixed_point_orthogonalize:
                # Make unit norm
                W = W / T.sqrt(T.sqr(W).sum(axis=0))
                # online approximate orthogonalization of columns
                for i in xrange(20):
                    W = 1.5*W - 0.5 * T.dot(W,T.dot(W.T,W))
                    W = W / T.sqrt(T.sqr(W).sum(axis=0))


            #WR = MRG_RandomStreams(42).normal(avg=0., size=W.shape, dtype=W.dtype)
            #WR.name = 'WR'
            #mask = beta_norms < 1e-3
            #mask.name = 'mask'
            #W = WR * mask + W * (1-mask)
            #W.name = 'W'

            if self.censor_beta_norms:
                beta_norms = T.sqrt( (T.sqr(W) * beta.dimshuffle(0, 'x')).sum(axis=0))
                W = W / beta_norms


            updates[self.W] = W

    def get_monitoring_channels(self, X, Y=None, **kwargs):

        return {
                'scale_min' : self.scale.min(),
                'scale_mean' : self.scale.mean(),
                'scale_max' : self.scale.max(),
                'beta_min' : self.beta.min(),
                'beta_mean' : self.beta.mean(),
                'beta_max' : self.beta.max()
                }

    def get_weights(self):
        return self.W.get_value()

    def get_weights_format(self):
        return ('v', 'h')

class OrthoRBM_NLL(Cost):

    supervised = False

    def __init__(self, constraint_coeff, use_admm=False):
        self.__dict__.update(locals())
        del self.self

    def __call__(self, model, X, Y=None, dual=None, **kwargs):

        assert (Y is None) == (not self.supervised)

        WBW = T.dot(model.W.T * model.beta,  model.W)
        target = T.identity_like(WBW)
        err = WBW - target
        penalty = T.sqr(err).sum()

        basic_cost = - model.log_likelihood(X).mean() + self.constraint_coeff * penalty

        if self.use_admm:
            if dual is None:
                if not hasattr(model, 'dual'):
                    model.dual = sharedX(np.zeros((model.nhid, model.nhid)), 'lambda')
                dual = model.dual
            augmented_lagrangian = basic_cost + (dual * err).sum()
            return augmented_lagrangian
        else:
            return basic_cost
        assert False # should be unreached

    def get_gradients(self, model, X, Y=None, **kwargs):

        assert 'dual' not in kwargs
        updates = {}

        if self.use_admm:
            rho = self.constraint_coeff * 2.
            dual = model.dual
            WBW = T.dot(model.W.T * model.beta, model.W)
            target = T.identity_like(WBW)
            err = WBW - target
            new_dual = dual + rho * err
            new_dual = block_gradient(new_dual)
            kwargs['dual'] = new_dual
            updates[dual] = new_dual

        cost = self(model, X, Y, **kwargs)

        params = model.get_params()
        assert not isinstance(params, set)
        return dict(zip(params, T.grad(cost, params))), updates


    def get_monitoring_channels(self, model, X, Y=None, **kwargs):

        if not self.supervised:
            Y = None

        WBW = T.dot(model.W.T * model.beta,  model.W)
        target = T.identity_like(WBW)
        err = WBW - target
        penalty = T.sqr(err).sum()
        log_likelihood =  model.log_likelihood(X).mean()

        diag = (T.sqr(model.W) * model.beta.dimshuffle(0,'x')).sum(axis=0)
        diag_penalty = T.sqr(diag-1.).sum()

        rval = {
                'constraint_sum_sq_err' : penalty,
                'diagonal_constraint_sum_sq_err' : diag_penalty,
                'log_likelihood' : log_likelihood }

        if self.use_admm:
            dual = model.dual
            rval['dual_min'] = dual.min()
            rval['dual_max'] = dual.max()
            rval['dual_mean'] = dual.mean()
            abs_dual = abs(dual)
            rval['abs_dual_min'] = abs_dual.min()
            rval['abs_dual_mean'] = abs_dual.mean()
            rval['abs_dual_max'] = abs_dual.max()

        return rval

class BetaOrthogonalize(TrainExtension):

    def __init__(self, run_freq):
        self.run_freq = run_freq
        self.count = 0

    def on_monitor(self, model, dataset, algorithm):
        self.count += 1
        if self.count % self.run_freq == 0:
            W = model.W.get_value()
            beta = model.beta.get_value()
            new_W = beta_orthogonalize(W, beta)
            model.W.set_value(new_W)

