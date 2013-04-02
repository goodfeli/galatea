from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial
from pylearn2.costs.cost import Cost
import numpy as np
import theano.tensor as T
from pylearn2.models.mlp import Sigmoid
from pylearn2.models.mlp import Layer
from pylearn2.space import VectorSpace
from theano.printing import Print

class Im2Word(DenseDesignMatrix):

    def __init__(self, start, stop):
        img = np.load('/data/lisatmp/goodfeli/esp/imgfeat.npy')
        word = serial.load("/data/lisatmp/goodfeli/esp_bow.pkl")
        self.words = word.words
        word = word.X

        img = img[start:stop, :]
        word = word[start:stop, :]

        super(Im2Word, self).__init__(X = img, y=word)


class FinalIm2Word(DenseDesignMatrix):

    def __init__(self, start=0, stop=1000):
        img = np.load('/data/lisatmp/goodfeli/esp/finalfeat.npy').astype('float32')
        word = serial.load("/data/lisatmp/goodfeli/final_bow.pkl")
        self.words = word.words
        word = word.X.astype('float32')
        img = img[start:stop,:]
        word = word[start:stop,:]

        super(FinalIm2Word, self).__init__(X = img, y=word)


class FinalSmallIm2Word(DenseDesignMatrix):

    def __init__(self):
        img = np.load('/data/lisatmp/goodfeli/esp/finalsmallfeat.npy').astype('float32')
        word = serial.load("/data/lisatmp/goodfeli/final_bow.pkl")
        self.words = word.words
        word = word.X.astype('float32')

        super(FinalSmallIm2Word, self).__init__(X = img, y=word)

class FinalTinyIm2Word(DenseDesignMatrix):

    def __init__(self):
        img = np.load('/data/lisatmp/goodfeli/esp/finaltinyfeat.npy').astype('float32')
        word = serial.load("/data/lisatmp/goodfeli/final_bow.pkl")
        self.words = word.words
        word = word.X.astype('float32')

        super(FinalSmallIm2Word, self).__init__(X = img, y=word)

class FinalMaxIm2Word(DenseDesignMatrix):

    def __init__(self):
        img = np.load('/data/lisatmp/goodfeli/esp/finalsmallfeat.npy').astype('float32')
        img2 = np.load('/data/lisatmp/goodfeli/esp/finalfeat.npy').astype('float32')
        img = np.maximum(img, img2)
        img2 = np.load('/data/lisatmp/goodfeli/esp/finaltinyfeat.npy').astype('float32')
        img = np.maximum(img, img2)
        word = serial.load("/data/lisatmp/goodfeli/final_bow.pkl")
        self.words = word.words
        word = word.X.astype('float32')

        super(FinalMaxIm2Word, self).__init__(X = img, y=word)



class NegF1(Cost):

    supervised = True

    def __call__(self, model, X, Y, **kwargs):

        Y_hat = model.fprop(X)

        expected_tp = (Y * Y_hat).sum()
        expected_pred_p = Y_hat.sum()

        precision = expected_tp / T.maximum(1e-7, expected_pred_p)

        recall = expected_tp / T.maximum(1e-7, Y.sum())

        F = 2. * precision * recall / T.maximum(1e-7, precision + recall)

        rval = -F

        rval.name = 'neg_f'

        return rval

class load_rbm(Sigmoid):

    def __init__(self, freeze=False, *args, **kwargs):
        self.freeze = freeze
        super(load_rbm, self).__init__(*args, **kwargs)

    def set_input_space(self, *args, **kwargs):
        super(load_rbm, self).set_input_space(*args, **kwargs)
        rbm = serial.load('/u/goodfeli/galatea/esp/sparse_rbm.pkl')

        rbm_W = rbm.get_weights()
        b = rbm.visible_layer.bias.get_value()

        lW, = self.transformer.get_params()
        lb = self.b

        lb.set_value(b)
        lW.set_value(rbm_W.T)

    def get_params(self):
        if self.freeze:
            return []
        return super(load_rbm, self).get_params()


class NegAveF1(Cost):

    supervised = True

    def __call__(self, model, X, Y, **kwargs):

        Y_hat = model.fprop(X)

        expected_tp = (Y * Y_hat).sum(axis=0)
        expected_pred_p = Y_hat.sum(axis=0)

        precision = expected_tp / T.maximum(1e-7, expected_pred_p)

        recall = expected_tp / T.maximum(1e-7, Y.sum(axis=0))

        F = 2. * precision * recall / T.maximum(1e-7, precision + recall)

        rval = -F.mean()

        rval.name = 'neg_f'

        return rval

class GlobalMax(Layer):

    def __init__(self, layer_name):
        self.__dict__.update(locals())
        del self.self
        self._params = []

    def set_input_space(self, space):
        self.input_space = space
        self.output_space = VectorSpace(space.num_channels)

    def fprop(self, state_below):
        return state_below.max(axis=2).max(axis=1).T
