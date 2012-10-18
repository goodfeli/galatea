from pylearn2.datasets.mnist import MNIST
from theano.printing import Print
import numpy as np
from pylearn2.utils import sharedX
from theano import tensor as T
from pylearn2.models.model import Model

class cDBM(Model):
    def __init__(self,W1, b1,W2,b2, W3, b3, mf_iter):
        self.mf_iter = mf_iter
        self.W1 = sharedX(W1)
        self.W2 = sharedX(W2)
        self.b1 = sharedX(b1)
        self.b2 = sharedX(b2)
        self.W3 = sharedX(W3)
        self.b3 = sharedX(b3)
        self.dataset_yaml_src = "!obj:pylearn2.datasets.mnist.MNIST { which_set : train }"

    def get_weights(self):
        return self.W1.get_value()

    def get_weights_format(self):
        return ('v','h')


    def params(self):
        return [ self.W1, self.W2, self.b1, self.b2 ]

    def mf1y(self, X):
        H1 = T.nnet.sigmoid(T.dot(X,self.W1)+self.b1)
        H2 = T.nnet.sigmoid(T.dot(H1,self.W2)+self.b2)
        y = T.nnet.softmax(T.dot(H2,self.W3)+self.b3)
        #y = Print('y')(y)
        return y

    def mf1y_arg(self, X):
        H1 = T.nnet.sigmoid(T.dot(X,self.W1)+self.b1)
        H2 = T.nnet.sigmoid(T.dot(H1,self.W2)+self.b2)
        return T.dot(H2,self.W3)+self.b3

    def mfny(self,X):
        H1 = T.nnet.sigmoid(T.dot(X,self.W1)+self.b1)
        H2 = T.nnet.sigmoid(T.dot(H1,self.W2)+self.b2)
        y = T.nnet.softmax(T.dot(H2,self.W3)+self.b3)

        for i in xrange(self.mf_iter-1):
            H1 = T.nnet.sigmoid(T.dot(X,self.W1)+T.dot(H2,self.W2.T)+self.b1)
            H2 = T.nnet.sigmoid(T.dot(H1,self.W2)+T.dot(y,self.W3.T)+self.b2)
            y = T.nnet.softmax(T.dot(H2,self.W3)+self.b3)
        return y

    def mfny_arg(self,X):
        H1 = T.nnet.sigmoid(T.dot(X,self.W1)+self.b1)
        H2 = T.nnet.sigmoid(T.dot(H1,self.W2)+self.b2)
        y = T.nnet.softmax(T.dot(H2,self.W3)+self.b3)

        for i in xrange(self.mf_iter-1):
            H1 = T.nnet.sigmoid(T.dot(X,self.W1)+T.dot(H2,self.W2.T)+self.b1)
            H2 = T.nnet.sigmoid(T.dot(H1,self.W2)+T.dot(y,self.W3.T)+self.b2)
            y = T.nnet.softmax(T.dot(H2,self.W3)+self.b3)
        return T.dot(H2,self.W3) + self.b3

