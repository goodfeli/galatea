import numpy as np
from pylearn2.utils import sharedX
from theano import config
from pylearn2.models.s3c import S3C
from pylearn2.models.s3c import SufficientStatistics
from theano import function
from theano.printing import Print

import theano.tensor as T

def clip(M):
    return np.clip(M,1e-7,1.-1e-7)

def norm_sq(s):
    return np.square(s.get_value()).sum()

def scale(s, a):
    s.set_value(s.get_value() * a)

class BatchGradientInference:
    """ A class for performing inference explicitly via optimization.
    Minimizes the KL divergence by adjusting the variational parameters.
    The method used is batch gradient descent with line searches. This
    method is slow but very exhaustive so it can be used to test other
    methods. """

    def __init__(self, model):
        """ model must be a PDDBM or S3C model """

        self.verbose = False
        batch_size = 87
        model._test_batch_size = batch_size

        self.model = model

        pddbm = hasattr(model,'dbm')


        if not pddbm:
            #hack s3c model to follow pddbm interface
            model.inference_procedure = model.e_step
            has_labels = False
        else:
            has_labels = model.dbm.num_classes > 0


        V = T.matrix("V")
        if has_labels:
            Y = T.matrix("Y")
        else:
            Y = None

        if config.compute_test_value != 'off':
            V.tag.test_value = np.cast[V.type.dtype](model.get_input_space().get_origin_batch(batch_size))

        self.model.make_pseudoparams()

        obs = {}
        for key in model.inference_procedure.hidden_obs:
            obs[key] = model.inference_procedure.hidden_obs[key]

        obs['H_hat'] = T.clip(obs['H_hat'],1e-7,1.-1e-7)
        if pddbm:
            obs['G_hat'] = tuple([ T.clip(elem,1e-7,1.-1e-7) for elem in obs['G_hat']  ])


        needed_stats = S3C.expected_log_prob_vhs_needed_stats()

        trunc_kl = model.inference_procedure.truncated_KL(V,  obs, Y).mean()

        assert len(trunc_kl.type.broadcastable) == 0

        if pddbm:
            G = model.inference_procedure.hidden_obs['G_hat']
            h_dim = model.s3c.nhid
        else:
            h_dim = model.nhid
        H = model.inference_procedure.hidden_obs['H_hat']
        S = model.inference_procedure.hidden_obs['H_hat']


        inputs = [ V ]
        if has_labels:
            inputs.append(Y)

        if self.verbose:
            print 'batch gradient class compiling init function'
        self.init_kl = function(inputs, trunc_kl)
        if self.verbose:
            print 'done'




        new_stats = SufficientStatistics.from_observations( needed_stats = needed_stats, V = V, H_hat = H, S_hat = S,
                    var_s0_hat = obs['var_s0_hat'], var_s1_hat = obs['var_s1_hat'])


        #obs = {
        #        "H_hat" : H,
        #        "S_hat" : S,
        #        "var_s0_hat" : obs['var_s0_hat'],
        #        "var_s1_hat" : obs['var_s1_hat'],
        #        }

        if pddbm:
            obs['G_hat'] = G

        obj = self.model.inference_procedure.truncated_KL( V, obs, Y ).mean()

        if pddbm:
            grad_G_sym = [ T.grad(obj, G_elem) for G_elem in G ]
        grad_H_sym = T.grad(obj,H)
        grad_S_sym = T.grad(obj,S)


        grad_H = sharedX( H.get_value())
        grad_S = sharedX( S.get_value())

        updates = { grad_H : grad_H_sym, grad_S : grad_S_sym }

        if pddbm:
            grad_G = [ sharedX( G_elem.get_value())  for G_elem in G ]
            for grad_G_elem, grad_G_sym_elem in zip(grad_G,grad_G_sym):
                updates[grad_G_elem] = grad_G_sym_elem

        if self.verbose:
            print 'batch gradient class compiling gradient function'
        self.compute_grad = function(inputs, updates = updates )
        if self.verbose:
            print 'done'



        if self.verbose:
            print 'batch gradient class compiling objective function'
        self.obj = function(inputs, obj)
        if self.verbose:
            print 'done'

        self.S = S
        self.H = H
        self.grad_S = grad_S
        self.grad_H = grad_H
        if pddbm:
            self.G = G
            self.grad_G = grad_G
        self.pddbm = pddbm
        self.has_labels = has_labels

    def cache_values(self):

        self.H_cache = self.H.get_value()
        self.S_cache = self.S.get_value()
        if self.pddbm:
            self.G_cache = [ G_elem.get_value() for G_elem in self.G ]

    def goto_alpha(self, a):

        assert not np.any(np.isnan(self.H_cache))
        piece_of_shit = self.grad_H.get_value()
        assert not np.any(np.isnan(piece_of_shit))
        if np.any(np.isinf(piece_of_shit)):
            print self.H.get_value()[np.isinf(piece_of_shit)]
            assert False
        mul = a * piece_of_shit

        assert not np.any(np.isnan(mul))

        diff = self.H_cache - mul

        assert not np.any(np.isnan(diff))

        fuck_you = clip( diff )

        assert not np.any(np.isnan(fuck_you))


        self.H.set_value(fuck_you)
        self.S.set_value(self.S_cache-a*self.grad_S.get_value())
        if self.pddbm:
            for G_elem, G_cache_elem, grad_G_elem in zip(self.G, self.G_cache, self.grad_G):
                G_elem.set_value(clip(G_cache_elem-a*grad_G_elem.get_value()))

    def normalize_grad(self):
        if self.pddbm:
            n = sum( [ norm_sq(elem) for elem in self.grad_G ] )
        else:
            n = 0.0
        n += norm_sq(self.grad_H)
        n += norm_sq(self.grad_S)

        n = np.sqrt(n)

        if self.pddbm:
            for elem in self.grad_G:
                scale(elem, 1./n)
        scale(self.grad_H, 1./n)
        scale(self.grad_S, 1./n)

    def run_inference(self, X, Y ):

        if (Y is not None) != (self.has_labels):
            print Y is not None
            print self.has_labels
            raise AssertionError()

        alpha_list = [ .001, .005, .01, .05, .1 ]

        self.model.inference_procedure.update_var_params(X,Y)

        assert self.H.get_value().shape[0] == X.shape[0]

        self.H.set_value(clip(self.H.get_value()))

        if self.has_labels:
            orig_kl = self.init_kl(X,Y)
        else:
            orig_kl = self.init_kl(X)

        if self.verbose:
            print orig_kl

        orig_H = self.H.get_value()
        assert orig_H.shape[0] == X.shape[0]
        orig_S = self.S.get_value()
        if self.pddbm:
            orig_G = [ G_elem.get_value() for G_elem in self.G ]

        while True:

            if self.has_labels:
                best_kl, best_alpha, best_alpha_ind = self.obj(X,Y), 0., -1
            else:
                best_kl, best_alpha, best_alpha_ind = self.obj(X), 0., -1
            #assert best_kl <= orig_kl
            self.cache_values()
            if self.has_labels:
                self.compute_grad(X,Y)
            else:
                self.compute_grad(X)
            self.normalize_grad()

            prev_best_kl = best_kl

            for ind, alpha in enumerate(alpha_list):
                self.goto_alpha(alpha)
                if self.has_labels:
                    kl = self.obj(X,Y)
                else:
                    kl = self.obj(X)
                if self.verbose:
                    print '\t',alpha,kl

                if kl < best_kl:
                    best_kl = kl
                    best_alpha = alpha
                    best_alpha_ind = ind

            if self.verbose:
                print best_kl

            assert not np.isnan(best_kl)
            #assert best_kl <= prev_best_kl
            self.goto_alpha(best_alpha)

            if best_alpha_ind < 1 and alpha_list[0] > 3e-7:
                alpha_list = [ alpha / 3. for alpha in alpha_list ]
            elif best_alpha_ind > len(alpha_list) -2:
                alpha_list = [ alpha * 2. for alpha in alpha_list ]
            elif best_alpha_ind == -1 and alpha_list[0] <= 3e-7:
                break


        rval =  { 'orig_S' : orig_S,
                'orig_H' : orig_H,
                'H' : self.H.get_value(),
                'S' : self.S.get_value(),
                'orig_kl' : orig_kl,
                'kl' : kl }

        if self.pddbm:
            rval['orig_G'] = orig_G
            rval['G'] =  [ G_elem.get_value() for G_elem in self.G ]

        return rval
