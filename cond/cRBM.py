from pylearn2.datasets.mnist import MNIST
from theano.printing import Print
import numpy as np
from pylearn2.utils import sharedX
from theano import tensor as T

class cRBM:
    def __init__(self,W1, b1,W2,b2, mf_iter):
        self.mf_iter = mf_iter
        self.W1 = sharedX(W1)
        self.W2 = sharedX(W2)
        self.b1 = sharedX(b1)
        self.b2 = sharedX(b2)
        self.dataset_yaml_src = "!obj:pylearn2.datasets.mnist.MNIST { which_set : train }"

    def get_weights(self):
        return self.W1.get_value()

    def get_weights_format(self):
        return ('v','h')


    def params(self):
        return [ self.W1, self.W2, self.b1, self.b2 ]

    def mf1y(self, X):
        H = T.nnet.sigmoid(T.dot(X,self.W1)+self.b1)
        y = T.nnet.softmax(T.dot(H,self.W2)+self.b2)
        #y = Print('y')(y)
        return y

    def mf1H(self, X):
        H = T.nnet.sigmoid(T.dot(X,self.W1)+self.b1)
        return H

    def mf1y_arg(self, X):
        H = T.nnet.sigmoid(T.dot(X,self.W1)+self.b1)
        return T.dot(H,self.W2)+self.b2

    def mfny(self,X):
        H = T.nnet.sigmoid(T.dot(X,2*self.W1)+self.b1)
        y = T.nnet.softmax(T.dot(H,self.W2)+self.b2)

        for i in xrange(self.mf_iter-1):
            H = T.nnet.sigmoid(T.dot(X,self.W1)+T.dot(y,self.W2.T)+self.b1)
            y = T.nnet.softmax(T.dot(H,self.W2)+self.b2)
        return y

    def mfnH(self,X):
        H = T.nnet.sigmoid(T.dot(X,2*self.W1)+self.b1)
        y = T.nnet.softmax(T.dot(H,self.W2)+self.b2)

        for i in xrange(self.mf_iter-1):
            H = T.nnet.sigmoid(T.dot(X,self.W1)+T.dot(y,self.W2.T)+self.b1)
            y = T.nnet.softmax(T.dot(H,self.W2)+self.b2)
        return H

    def mfny_arg(self,X):
        H = T.nnet.sigmoid(T.dot(X,2*self.W1)+self.b1)
        y = T.nnet.softmax(T.dot(H,self.W2)+self.b2)

        for i in xrange(self.mf_iter-1):
            H = T.nnet.sigmoid(T.dot(X,self.W1)+T.dot(y,self.W2.T)+self.b1)
            y = T.nnet.softmax(T.dot(H,self.W2)+self.b2)
        return T.dot(H,self.W2) + self.b2

