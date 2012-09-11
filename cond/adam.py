import numpy as np
from galatea.dbm.inpaint.super_dbm import GaussianConvolutionalVisLayer
from galatea.dbm.inpaint.super_dbm import Softmax
from pylearn2.linear.conv2d import make_random_conv2D
from pylearn2.utils import sharedX
from pylearn2.space import Conv2DSpace
import theano.tensor as T
from theano.tensor.signal.downsample import max_pool_2d
import warnings
warnings.warn("this doesn't actually implement adam's pipeline, it lacks the preprocessing")
from galatea.cond.neighbs import cifar10neighbs
from galatea.cond.neighbs import multichannel_neibs2imgs
from pylearn2.linear.matrixmul import MatrixMul
from theano.printing import Print

class Adam:
    def __init__(self, batch_size, alpha, irange):
        self.alpha = alpha
        self.visible_layer = GaussianConvolutionalVisLayer(rows = 32,cols = 32, channels = 3, init_beta =1., init_mu = 0.)
        self.hidden_layers = [ Softmax(n_classes = 10,
                                            irange = .01) ]
        rng = np.random.RandomState([2012,8,20])
        self.W = MatrixMul( sharedX( rng.uniform(-irange, irange, (108,1600))))
        #make_random_conv2D(irange = .05, input_space = self.visible_layer.get_input_space(),
        #                output_space = Conv2DSpace([27,27],1600),
        #                kernel_shape = (6,6),
        #                batch_size = batch_size)
        self.batch_size = batch_size
        self.hidden_layers[0].dbm = self
        self.hidden_layers[0].set_input_space(Conv2DSpace([2,2],3200))

    def get_params(self):
        return set(self.hidden_layers[0].get_params()).union(self.W.get_params())

    def mf(self, X):
        patches = cifar10neighbs(X,(6,6))
        patches -= patches.mean(axis=1).dimshuffle(0,'x')
        patches /= T.sqrt(T.sqr(patches).sum(axis=1)+10.0).dimshuffle(0,'x')

        Z = self.W.lmul(patches)

        #Z = Print('Z',attrs=['min','mean','max'])(Z)

        Z = T.concatenate((Z,-Z),axis=1)
        Z = multichannel_neibs2imgs(Z, self.batch_size, 27, 27, 3200, 1, 1)
        Z = Z.dimshuffle(0,3,1,2)
        p = max_pool_2d(Z,(14,14),False)
        p = p.dimshuffle(0,1,2,3)
        p = T.maximum(p - self.alpha, 0.)
        #p = Print('p',attrs=['min','mean','max'])(p)
        y = self.hidden_layers[0].mf_update(state_below = p, state_above = None)
        return [ Z, y ]

    def get_weights_topo(self):
        outp, inp, rows, cols = range(4)
        raw = self.W._filters.get_value()
        return np.transpose(raw,(outp,rows,cols,inp))
