

import sys
import numpy as np
from pylearn2.utils import sharedX
from theano import config
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from pylearn2.models.s3c import S3C
from pylearn2.models.s3c import SufficientStatistics
from theano import function

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
        """ model must be a PDDBM.
            other models like S3C could be supported in principle but aren't yet."""

        self.model = model

        assert hasattr(model,'dbm')

        V = T.matrix("V")

        self.model.make_pseudoparams()

        obs = model.inference_procedure.infer(V)
        obs['H_hat'] = T.clip(obs['H_hat'],1e-7,1.-1e-7)
        obs['G_hat'] = tuple([ T.clip(elem,1e-7,1.-1e-7) for elem in obs['G_hat']  ])


        needed_stats = S3C.expected_log_prob_vhs_needed_stats()

        trunc_kl = model.inference_procedure.truncated_KL(V, obs).mean()

        assert len(trunc_kl.type.broadcastable) == 0

        batch_size = 1

        G = [ sharedX(np.zeros((batch_size, rbm.nhid), dtype='float32')) for rbm in model.dbm.rbms ]
        H = sharedX(np.zeros((batch_size, model.s3c.nhid), dtype='float32'))
        S = sharedX(np.zeros((batch_size, model.s3c.nhid), dtype='float32'))

        new_stats = SufficientStatistics.from_observations( needed_stats = needed_stats, V = V, H_hat = H, S_hat = S,
                    var_s0_hat = obs['var_s0_hat'], var_s1_hat = obs['var_s1_hat'])


        obj = self.model.inference_procedure.truncated_KL( V, {
                "H_hat" : H,
                "S_hat" : S,
                "var_s0_hat" : obs['var_s0_hat'],
                "var_s1_hat" : obs['var_s1_hat'],
                "G_hat" : G
                } ).mean()

        grad_G_sym = [ T.grad(obj, G_elem) for G_elem in G ]
        grad_H_sym = T.grad(obj,H)
        grad_S_sym = T.grad(obj,S)


        grad_H = sharedX( H.get_value())
        grad_S = sharedX( S.get_value())
        grad_G = [ sharedX( G_elem.get_value())  for G_elem in G ]

        updates = { grad_H : grad_H_sym, grad_S : grad_S_sym }

        for grad_G_elem, grad_G_sym_elem in zip(grad_G,grad_G_sym):
            updates[grad_G_elem] = grad_G_sym_elem

        print 'batch gradient class compiling gradient function'
        self.compute_grad = function([V], updates = updates )
        print 'done'

        updates = { H : obs['H_hat'], S : obs['S_hat'] }

        for G_elem, G_hat_elem in zip(G, obs['G_hat']):
            updates[G_elem] = G_hat_elem

        print 'batch gradient class compiling init function'
        self.init = function([V], trunc_kl,  updates = updates )
        print 'done'

        print 'batch gradient class compiling objective function'
        self.obj = function([V], obj)
        print 'done'

        self.S = S
        self.H = H
        self.G = G
        self.grad_S = grad_S
        self.grad_H = grad_H
        self.grad_G = grad_G

    def cache_values(self):

        self.H_cache = self.H.get_value()
        self.S_cache = self.S.get_value()
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
        for G_elem, G_cache_elem, grad_G_elem in zip(self.G, self.G_cache, self.grad_G):
            G_elem.set_value(clip(G_cache_elem-a*grad_G_elem.get_value()))

    def normalize_grad(self):
        n = sum( [ norm_sq(elem) for elem in self.grad_G ] )
        n += norm_sq(self.grad_H)
        n += norm_sq(self.grad_S)

        n = np.sqrt(n)

        for elem in self.grad_G:
            scale(elem, 1./n)
        scale(self.grad_H, 1./n)
        scale(self.grad_S, 1./n)

    def run_inference(self, X):

        alpha_list = [ .001, .005, .01, .05, .1 ]

        orig_kl = self.init(X)

        self.H.set_value(clip(self.H.get_value()))

        print orig_kl

        orig_H = self.H.get_value()
        orig_S = self.S.get_value()
        orig_G = [ G_elem.get_value() for G_elem in self.G ]

        while True:
            best_kl, best_alpha, best_alpha_ind = self.obj(X), 0., -1

            self.cache_values()
            self.compute_grad(X)
            self.normalize_grad()

            for ind, alpha in enumerate(alpha_list):
                self.goto_alpha(alpha)
                kl = self.obj(X)
                print '\t',alpha,kl

                if kl < best_kl:
                    best_kl = kl
                    best_alpha = alpha
                    best_alpha_ind = ind

            print best_kl
            assert not np.isnan(best_kl)
            self.goto_alpha(best_alpha)

            if best_alpha_ind < 1 and alpha_list[0] > 3e-7:
                alpha_list = [ alpha / 3. for alpha in alpha_list ]
            elif best_alpha_ind > len(alpha_list) -2:
                alpha_list = [ alpha * 2. for alpha in alpha_list ]
            elif best_alpha_ind == -1 and alpha_list[0] <= 3e-7:
                break


        return { 'orig_S' : orig_S,
                'orig_H' : orig_H,
                'orig_G' : orig_G,
                'G' : [ G_elem.get_value() for G_elem in self.G ],
                'H' : self.H.get_value(),
                'S' : self.S.get_value(),
                'orig_kl' : orig_kl,
                'kl' : kl }

