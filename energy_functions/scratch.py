""" A bunch of random energy functions that are fairly experimental """
import numpy as N
from theano import config
floatX = config.floatX
from theano import shared
import theano.tensor as T
from .energy_function import EnergyFunction
from theano.printing import Print

class recons_model_1(EnergyFunction):
    """
        E(X) = mse_vis_prec(X,recons(X)) - dot(delta,h(X))

        where h(X) = sigmoid(dot(X,W)+b_h)
              recons(X) = dot(h(X),W.T)+b_v
              mse_vis_prec(X,Y) = mean_i ( mean_j ( vis_prec_j (X_ij - Y_ij)^2 ) )
    """

    def __init__(self,
            nvis, init_bias_hid, nhid, vis_prec_lr_scale, init_vis_prec, irange, learn_vis_prec, init_delta):

        super(recons_model_1,self).__init__()

        self.nvis = nvis
        self.init_bias_hid = init_bias_hid
        self.nhid = nhid

        self.vis_prec_lr_scale = vis_prec_lr_scale
        self.init_vis_prec = init_vis_prec

        self.irange = irange

        self.learn_vis_prec = learn_vis_prec
        self.init_delta = init_delta

        self.reset_rng()

        self.redo_everything()
    #

    def reset_rng(self):
        self.rng = N.random.RandomState([1,2,3])
    #

    def redo_everything(self):
        self.W = shared( N.cast[floatX](self.rng.uniform(-self.irange,self.irange,(self.nvis,self.nhid))),
                        name = 'W')
        self.bias_hid = shared( N.zeros((self.nhid,),dtype=floatX)+self.init_bias_hid, name='bias_hid')
        self.bias_vis = shared( N.zeros((self.nvis,),dtype=floatX), name='bias_vis')

        self.delta = shared( N.zeros((self.nhid,),dtype=floatX)+self.init_delta, name='delta')

        self.params = [ self.W , self.bias_hid, self.bias_vis, self.delta ]

        self.vis_prec_driver = shared(N.zeros((self.nvis,),dtype=floatX) + self.init_vis_prec, name='vis_prec')

        if self.learn_vis_prec:
            self.params.append(self.vis_prec_driver)
        #
    #

    def get_weights_format(self):
        return ['v','h']
    #

    def get_weights(self, borrow = False):
        return self.W.get_value(borrow = borrow)
    #

    def encode(self, X):
        X_name = 'X' if X.name is None else X.name
        H = T.nnet.sigmoid(T.dot(X, self.W) + self.bias_hid)
        H.name = 'H('+X_name+')'
        return H
    #

    def decode(self, H):
        H_name = 'H' if H.name is None else H.name
        R = T.dot(H, self.W.T) + self.bias_vis
        R.name = 'R('+H_name+')'
        return R
    #

    def __call__(self, X):
        #temp = self.W
        #self.W = Print('W',attrs=['min','max'])(self.W)

        X_name = 'X' if X.name is None else X.name
        H = self.encode(X)
        R = self.decode(H)
        diff = X-R
        diff.name = 'scratch_recons_model_1_diff('+X_name+')'
        sqdiff = T.sqr(diff)
        sqdiff.name = 'scratch_recons_model_1_sqdiff('+X_name+')'
        vis_prec = self.vis_prec_driver * self.vis_prec_lr_scale
        #TODO: probably faster as tensordot, but there is no tensordot on gpu
        wdiff = vis_prec * sqdiff
        wdiff.name = 'scratch_recons_model_1_wdiff('+X_name+')'
        recons_err = T.sum(wdiff,axis=1)
        recons_err.name = 'recons_err('+X_name+')'
        assert len(recons_err.type.broadcastable) == 1

        activation_bias = - T.sum(self.delta * H, axis=1)
        activation_bias.name = 'activ_bias('+X_name+')'
        assert len(activation_bias.type.broadcastable) == 1

        E = recons_err + activation_bias
        E.name = 'recons_model_1_E('+X_name+')'

        #self.W = temp

        return E
    #

    def censor_updates(self, updates):
        if self.vis_prec_driver in updates:
            assert self.learn_vis_prec

            updates[self.vis_prec_driver] = T.clip(updates[self.vis_prec_driver],1e-3/self.vis_prec_lr_scale,1e3/self.vis_prec_lr_scale)
        #
    #

    def get_params(self):
        return [ param for param in self.params ]
    #
#
